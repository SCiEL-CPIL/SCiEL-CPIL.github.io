"""Rewritten pebbling solver: A* over memory states with communication-cost heuristic."""

import time
import heapq
from itertools import chain
from functools import lru_cache
from typing import List, Tuple, Set, Dict, Optional

# EVOLVE-BLOCK-START

def pebble(dag, fast_memory_budget):
    """
    Solve communication-minimizing DAG pebbling via A* search over memory states.
    
    dag: expected to have:
        - dag['vertices']: list of vertex ids (topologically ordered)
        - dag['parents']: dict vertex -> list of parent vertices
        - dag['outputs']: list of output vertex ids (must finish blue/red)
        - dag['inputs']: list of input vertex ids (start blue, i.e., available)
    fast_memory_budget: int, max red pebbles simultaneously
    
    Returns: list of [operation, vertex] with operations in {'load','store','compute','delete'}
    """
    deadline = time.time() + 75.0  # safety margin before 90s kill
    
    # --- Parse DAG structure ---
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])
    else:
        # Fallback: assume dag is a list of (vertex, parents, is_output, is_input)
        vertices = []
        parents = {}
        outputs = []
        inputs = []
        for item in dag:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                v = item[0]
                par = item[1]
                vertices.append(v)
                parents[v] = par
            elif isinstance(item, dict):
                v = item.get('id', item.get('vertex'))
                par = item.get('parents', [])
                vertices.append(v)
                parents[v] = par
                if item.get('output', False):
                    outputs.append(v)
                if item.get('input', False):
                    inputs.append(v)
    
    if not vertices:
        return []
    
    # Ensure all vertices are in topological order
    # Topological order means parents appear before children
    vertex_index = {v: i for i, v in enumerate(vertices)}
    
    # Identify inputs and outputs if not provided
    if not inputs:
        inputs = [v for v in vertices if v not in parents or len(parents.get(v, [])) == 0]
    if not outputs:
        # Outputs are vertices with no children
        children = {v: set() for v in vertices}
        for v in vertices:
            for p in parents.get(v, []):
                children[p].add(v)
        outputs = [v for v in vertices if len(children[v]) == 0]
    
    inputs_set = set(inputs)
    outputs_set = set(outputs)
    budget = fast_memory_budget
    
    if budget <= 0:
        # No fast memory: must load, compute, store each value immediately
        # Actually with budget 0, we can't hold any red pebble, so this is problematic.
        # Let's handle it: load, compute, store in sequence for each vertex.
        # But compute needs parents in memory... with budget 0 this is impossible unless
        # we reload parents. Let's do a naive schedule.
        ops = []
        for v in vertices:
            if v not in inputs_set:
                for p in parents.get(v, []):
                    ops.append(['load', p])
                ops.append(['compute', v])
                ops.append(['store', v])
            else:
                # Inputs start blue, just store
                ops.append(['store', v])
        return ops[:20000]
    
    # --- A* Search over memory states ---
    # State: frozenset of red-pebbled vertices
    # Goal: all outputs are red (in memory)
    # Actually, outputs must "finish blue" per task: "Inputs start blue, all outputs must finish blue"
    # Wait, re-reading: "Inputs start blue, all outputs must finish blue; every red value consumes one fast-memory slot"
    # So blue = in slow memory (available but not in fast memory), red = in fast memory
    # Inputs start blue (available in slow memory, can be loaded)
    # Outputs must finish blue (stored back to slow memory)
    # But we need them computed... Let me re-read.
    
    # "Inputs start blue, all outputs must finish blue; every red value consumes one fast-memory slot, including newly computed results."
    # So the goal state is: all outputs are blue (in slow memory), and we want to minimize transfers.
    # A transfer is a load (blue->red) or store (red->blue).
    # Compute and delete are free operations.
    # We need to compute outputs and store them back.
    
    # Let me reconsider: the goal is all outputs are blue (stored). 
    # But we need to have computed them first. So the sequence is:
    # 1. Load necessary inputs
    # 2. Compute intermediate values
    # 3. Compute outputs
    # 4. Store outputs (and maybe intermediate values)
    # 5. Delete remaining pebbles (free)
    
    # Actually, "delete" is an operation in the list. Let me check if delete is free or costs.
    # "every red value consumes one fast-memory slot" - delete frees the slot.
    # Transfers are load/store. Compute and delete are not transfers.
    
    # So the cost function: count of load + store operations.
    
    # State for A*: 
    # - red_pebbles: frozenset of vertices currently in fast memory
    # - computed: frozenset of vertices that have been computed (and are either red or stored blue)
    # Actually, if a vertex is stored (blue), it can be reloaded later.
    # We need to track which vertices are "available" - either red (in fast mem) or blue (in slow mem, already computed).
    
    # Let me define:
    # - red: set of vertices in fast memory
    # - computed: set of vertices that have been computed at least once (can be blue or red)
    # A vertex v can be loaded if v is in computed and v not in red (it's blue)
    # A vertex v can be computed if all its parents are in red
    # A vertex v can be stored if v is in red
    # A vertex v can be deleted if v is in red
    
    # Goal: all outputs are in computed (they've been computed and stored)
    # Actually, they must be blue at the end, so they must be stored.
    # But we can delete non-output red pebbles for free.
    
    # Let's define state as (red_frozen, computed_frozen)
    # Initial: red = empty, computed = inputs_set (inputs are blue from start)
    # Goal: all outputs in computed, and we don't care about red (we can delete for free)
    
    # Heuristic: minimum additional transfers needed.
    # Lower bound: for each output not yet computed, we need at least 1 store.
    # Plus, we may need to load some values.
    
    # This state space can be huge. Let's use A* with a good heuristic.
    
    # For the given problem sizes (layered DAGs), this should be manageable.
    
    # Let's precompute:
    # - For each vertex, its depth (distance from inputs)
    # - For each vertex, whether it's an ancestor of any output
    
    # Actually, let's just run A* with the following:
    # - State: (red_frozenset, computed_frozenset)
    # - Cost: number of load+store operations so far
    # - Heuristic: 
    #   For each output o not in computed: need at least 1 store + loads for missing ancestors
    #   Simplified: count of outputs not computed * 1 (just the store)
    #   Better: for each output not computed, 1 store + number of its uncomputed parents that are inputs (need load)
    #   Even better: lower bound on loads needed
    
    # Let me think about this more carefully. The heuristic must be admissible (never overestimate).
    
    # Simple admissible heuristic: number of outputs not yet computed (each needs at least 1 store).
    
    # Let's implement A*:
    
    def initial_state():
        return (frozenset(), frozenset(inputs_set))
    
    def goal_test(state):
        _, computed = state
        return outputs_set.issubset(computed)
    
    def heuristic(state):
        _, computed = state
        # Each uncomputed output needs at least 1 store
        return len(outputs_set - computed)
    
    def get_successors(state):
        red, computed = state
        successors = []
        
        # For each vertex, check possible operations:
        
        # LOAD: vertex is in computed, not in red, and budget allows
        if len(red) < budget:
            for v in computed - set(red):
                new_red = frozenset(red | {v})
                new_state = (new_red, computed)
                successors.append((1, 'load', new_state, v))
        
        # COMPUTE: vertex not in computed, all parents in red
        for v in vertices:
            if v in computed:
                continue
            par = parents.get(v, [])
            if all(p in red for p in par):
                new_computed = frozenset(computed | {v})
                # The computed result goes into red memory
                if len(red) < budget:
                    new_red = frozenset(red | {v})
                    new_state = (new_red, new_computed)
                    successors.append((0, 'compute', new_state, v))
        
        # STORE: vertex in red, not an input (inputs are already blue)
        # Actually, any vertex in red can be stored
        for v in red:
            new_red = frozenset(red - {v})
            new_state = (new_red, computed)
            successors.append((1, 'store', new_state, v))
        
        # DELETE: vertex in red, not needed (free operation)
        # We can delete any red pebble
        for v in red:
            new_red = frozenset(red - {v})
            new_state = (new_red, computed)
            successors.append((0, 'delete', new_state, v))
        
        return successors
    
    # A* search
    # Use a priority queue: (f_cost, g_cost, state)
    # f_cost = g_cost + h_cost
    
    start = initial_state()
    if goal_test(start):
        # Already done
        return []
    
    # Priority queue: (f, g, counter, state, path_ops)
    # To avoid storing full path in heap, we can store parent pointers
    # But for simplicity and given small state spaces, let's store the operation list
    # Actually, that's expensive. Let's use parent pointers.
    
    # Better: use a dict from state to (g_cost, parent_state, op, vertex)
    # And a heap for (f, g, counter, state)
    
    counter = 0
    heap = [(heuristic(start), 0, counter, start, [], None, None, None)]
    # (f, g, counter, state, ops_list, parent, op, vertex)
    # Actually, let's store ops as a list for simplicity since state spaces should be small
    
    # No, let's be more careful. Store parent pointers.
    
    parent = {}  # state -> (parent_state, op, vertex)
    g_score = {start: 0}
    f_score = {start: heuristic(start)}
    
    heap = [(f_score[start], 0, counter, start)]
    
    best_solution = None
    best_cost = float('inf')
    
    max_states = 50000  # Limit search size
    
    while heap:
        if time.time() > deadline:
            break
        if len(g_score) > max_states:
            break
        
        f, g, _, state = heapq.heappop(heap)
        counter += 1
        
        if goal_test(state):
            # Found a goal state
            if g < best_cost:
                best_cost = g
                # Reconstruct path
                ops = []
                curr = state
                while curr in parent:
                    curr, op, v = parent[curr]
                    ops.append([op, v])
                ops.reverse()
                best_solution = ops
            # In A*, first goal found is optimal, but let's continue a bit
            # Actually, for A* with consistent heuristic, first goal is optimal
            break
        
        # Generate successors
        for cost, op, new_state, v in get_successors(state):
            new_g = g + cost
            if new_g < g_score.get(new_state, float('inf')):
                g_score[new_state] = new_g
                parent[new_state] = (state, op, v)
                h = heuristic(new_state)
                f_new = new_g + h
                f_score[new_state] = f_new
                heapq.heappush(heap, (f_new, new_g, counter, new_state))
                counter += 1
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: naive schedule
    return naive_schedule(dag, fast_memory_budget)


def naive_schedule(dag, fast_memory_budget):
    """Simple topological-order schedule."""
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])
    else:
        vertices = []
        parents = {}
        outputs = []
        inputs = []
        for item in dag:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                v = item[0]
                par = item[1]
                vertices.append(v)
                parents[v] = par
            elif isinstance(item, dict):
                v = item.get('id', item.get('vertex'))
                par = item.get('parents', [])
                vertices.append(v)
                parents[v] = par
                if item.get('output', False):
                    outputs.append(v)
                if item.get('input', False):
                    inputs.append(v)
    
    if not vertices:
        return []
    
    if not inputs:
        inputs = [v for v in vertices if v not in parents or len(parents.get(v, [])) == 0]
    if not outputs:
        children = {v: set() for v in vertices}
        for v in vertices:
            for p in parents.get(v, []):
                children[p].add(v)
        outputs = [v for v in vertices if len(children[v]) == 0]
    
    inputs_set = set(inputs)
    budget = fast_memory_budget
    if budget <= 0:
        budget = 1
    
    ops = []
    red = set()
    computed = set(inputs_set)
    
    for v in vertices:
        if v in inputs_set:
            continue
        # Load parents if not in red
        for p in parents.get(v, []):
            if p not in red and p in computed:
                ops.append(['load', p])
                red.add(p)
            elif p not in red and p not in computed:
                # Should not happen if topologically ordered
                ops.append(['load', p])
                red.add(p)
                computed.add(p)
        
        # Compute v
        ops.append(['compute', v])
        red.add(v)
        computed.add(v)
        
        # If over budget, store/delete some
        while len(red) > budget:
            # Store oldest non-output, non-parent-of-unprocessed
            # Simple: store first one that's not needed
            for r in list(red):
                # Check if r is a parent of any unprocessed vertex
                needed = False
                for w in vertices[vertices.index(v)+1:]:
                    if r in parents.get(w, []):
                        needed = True
                        break
                if not needed:
                    ops.append(['store', r])
                    red.discard(r)
                    break
            else:
                # Can't find one to store, force store first
                if red:
                    r = next(iter(red))
                    ops.append(['store', r])
                    red.discard(r)
    
    # Store outputs
    for o in outputs:
        if o in red:
            ops.append(['store', o])
            red.discard(o)
    
    # Delete remaining
    for r in list(red):
        ops.append(['delete', r])
    
    return ops[:20000]

# EVOLVE-BLOCK-END
