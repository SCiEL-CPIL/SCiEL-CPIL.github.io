"""Rewritten program for Communication-minimizing DAG pebbling."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Minimize communication (load/store) operations for DAG pebbling.
    Uses a time-reversed unpebbling heuristic: process from outputs to inputs,
    prioritizing deletion of vertices with highest in-degree among unprocessed
    predecessors to minimize peak live values and reduce store/reload cycles.
    Returns: list of [operation, vertex] where operation in {load, store, compute, delete}
    """
    deadline = time.time() + 50  # Safety margin before 60s limit

    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('nodes', dag.get('V', None)))
        parents = dag.get('parents', dag.get('edges', dag.get('pred', None)))
        inputs = dag.get('inputs', dag.get('sources', dag.get('input_nodes', None)))
        outputs = dag.get('outputs', dag.get('sinks', dag.get('output_nodes', None)))
    else:
        vertices = None
        parents = None
        inputs = None
        outputs = None

    if vertices is None or parents is None:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []

    # Normalize to sets/lists
    if isinstance(vertices, (list, tuple, range)):
        vertex_list = list(vertices)
    elif isinstance(vertices, int):
        vertex_list = list(range(vertices))
    else:
        vertex_list = list(vertices)

    num_vertices = len(vertex_list)

    # Normalize parents to dict
    if isinstance(parents, dict):
        parent_dict = dict(parents)
    elif isinstance(parents, (list, tuple)):
        parent_dict = {i: list(p) for i, p in enumerate(parents)}
    else:
        parent_dict = {}

    for v in vertex_list:
        if v not in parent_dict:
            parent_dict[v] = []

    # Normalize inputs and outputs
    if inputs is None:
        inputs = set(v for v in vertex_list if not parent_dict.get(v, []))
    elif isinstance(inputs, (set, list, tuple)):
        inputs = set(inputs)
    else:
        inputs = set(inputs)

    if outputs is None:
        children = {v: [] for v in vertex_list}
        for v in vertex_list:
            for p in parent_dict.get(v, []):
                children[p].append(v)
        outputs = set(v for v in vertex_list if not children[v])
    elif isinstance(outputs, (set, list, tuple)):
        outputs = set(outputs)
    else:
        outputs = set(outputs)

    # Build children map
    children = {v: [] for v in vertex_list}
    for v in vertex_list:
        for p in parent_dict.get(v, []):
            children[p].append(v)

    # Determine needed vertices (ancestors of outputs)
    needed = set(outputs)
    for v in reversed(vertex_list):
        if v in needed:
            for p in parent_dict.get(v, []):
                needed.add(p)

    needed_vertices = [v for v in vertex_list if v in needed]
    needed_set = set(needed_vertices)
    vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}

    # Precompute children within needed set
    needed_children = {v: [c for c in children.get(v, []) if c in needed_set] for v in needed_set}
    
    # Precompute parent list within needed set
    needed_parents = {v: [p for p in parent_dict.get(v, []) if p in needed_set] for v in needed_set}

    # === TIME-REVERSED UNPEBBLING HEURISTIC ===
    # Process from outputs backward to inputs.
    # At each step, choose which vertex to "unpebble" (remove from fast memory)
    # based on the number of unprocessed children (dependents) that still need it.
    # Vertices with fewer remaining dependents can be safely evicted sooner.
    # Vertices with many remaining dependents should be kept longer.
    
    # State for the reversed process
    in_fast = set()
    in_slow = set(inputs)  # Inputs start in slow memory
    computed = set()
    operations = []
    
    # Track which vertices have been "processed" (their children all handled)
    processed = set()
    # remaining_dependents[v] = number of children of v that haven't been processed yet
    remaining_dependents = {v: len(needed_children[v]) for v in needed_set}
    
    # Priority queue: vertices ready to be unpebbled (all children processed)
    # We process in reverse topological order
    # A vertex is ready when all its children have been processed
    
    ready = deque()
    for v in reversed(needed_vertices):
        # Check if all children are already in processed (or not needed)
        # Since we go in reverse topo order, children come before parents
        # So when we reach v, all its children should already be processed
        ready.append(v)
    
    # Process in reverse topological order
    for v in reversed(needed_vertices):
        if time.time() > deadline:
            break
        
        # v is being "unpebbled" - we decide if it should stay in fast or be evicted
        # First, mark v as processed
        processed.add(v)
        
        # For each parent of v, decrement remaining_dependents
        for p in needed_parents.get(v, []):
            remaining_dependents[p] -= 1
        
        # Decision: should v be in fast memory at this point?
        # v must be in fast if:
        #   - v is an output (must finish blue)
        #   - v has remaining dependents that need it soon
        # v can be evicted if:
        #   - v has no remaining dependents
        #   - v is not an output
        #   - fast memory is full and we need space
        
        if v in outputs:
            # Outputs must be in fast memory at the end
            # Ensure v is in fast
            if v not in in_fast:
                if len(in_fast) >= fast_memory_budget:
                    # Need to evict something
                    _evict_for_unpebbling(in_fast, in_slow, operations, remaining_dependents, 
                                           processed, needed_set, outputs, v)
                operations.append(['load', v])
                in_fast.add(v)
        else:
            # Non-output: decide based on remaining dependents
            rem = remaining_dependents[v]
            if rem == 0:
                # No more dependents need this - can evict
                if v in in_fast:
                    if v in in_slow:
                        operations.append(['delete', v])
                    else:
                        operations.append(['store', v])
                    in_fast.discard(v)
            else:
                # Still needed by dependents
                # Check if it's in fast; if not and we have room, load it
                # If fast is full, we may need to evict something else
                if v not in in_fast:
                    if len(in_fast) < fast_memory_budget:
                        operations.append(['load', v])
                        in_fast.add(v)
                    else:
                        # Fast is full - evict something with fewer remaining dependents
                        _evict_for_unpebbling(in_fast, in_slow, operations, remaining_dependents,
                                              processed, needed_set, outputs, v)
                        operations.append(['load', v])
                        in_fast.add(v)
        
        # If v is not an input, it was computed - mark as computed
        if v not in inputs:
            computed.add(v)
    
    # Ensure all outputs are in fast memory at the end
    for v in outputs:
        if v in needed_set and v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                _evict_for_unpebbling(in_fast, in_slow, operations, remaining_dependents,
                                      processed, needed_set, outputs, v)
            operations.append(['load', v])
            in_fast.add(v)
    
    # Final verification: all outputs must be in fast
    for v in outputs:
        if v in needed_set and v not in in_fast:
            if len(in_fast) < fast_memory_budget:
                operations.append(['load', v])
                in_fast.add(v)
            else:
                # Emergency: evict anything
                if in_fast:
                    u = next(iter(in_fast))
                    if u in in_slow:
                        operations.append(['delete', u])
                    else:
                        operations.append(['store', u])
                    in_fast.discard(u)
                    operations.append(['load', v])
                    in_fast.add(v)
    
    # Check operation limit
    if len(operations) > 20000:
        # Fall back to forward pass if we exceed limit
        return _forward_pass(dag, fast_memory_budget, deadline)
    
    return operations


def _evict_for_unpebbling(in_fast, in_slow, operations, remaining_dependents,
                          processed, needed_set, outputs, protect_v):
    """Evict a vertex from fast memory to make room.
    Prioritize evicting vertices with the fewest remaining dependents
    (they are least likely to be needed again)."""
    if not in_fast:
        return
    
    best_evict = None
    best_score = float('inf')
    
    for u in list(in_fast):
        if u == protect_v:
            continue
        if u in outputs:
            continue  # Don't evict outputs
        
        score = remaining_dependents.get(u, 0)
        # Prefer evicting vertices with fewer remaining dependents
        if score < best_score:
            best_score = score
            best_evict = u
    
    if best_evict is None:
        # All are outputs or protected - pick any non-protect
        for u in in_fast:
            if u != protect_v:
                best_evict = u
                break
    
    if best_evict is not None:
        if best_evict in in_slow:
            operations.append(['delete', best_evict])
        else:
            operations.append(['store', best_evict])
        in_fast.discard(best_evict)


def _forward_pass(dag, fast_memory_budget, deadline):
    """Fallback: original forward topological pass with FNU eviction."""
    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('nodes', dag.get('V', None)))
        parents = dag.get('parents', dag.get('edges', dag.get('pred', None)))
        inputs = dag.get('inputs', dag.get('sources', dag.get('input_nodes', None)))
        outputs = dag.get('outputs', dag.get('sinks', dag.get('output_nodes', None)))
    else:
        vertices = None
        parents = None
        inputs = None
        outputs = None

    if vertices is None or parents is None:
        return []

    # Normalize to sets/lists
    if isinstance(vertices, (list, tuple, range)):
        vertex_list = list(vertices)
    elif isinstance(vertices, int):
        vertex_list = list(range(vertices))
    else:
        vertex_list = list(vertices)

    # Normalize parents to dict
    if isinstance(parents, dict):
        parent_dict = dict(parents)
    elif isinstance(parents, (list, tuple)):
        parent_dict = {i: list(p) for i, p in enumerate(parents)}
    else:
        parent_dict = {}

    for v in vertex_list:
        if v not in parent_dict:
            parent_dict[v] = []

    # Normalize inputs and outputs
    if inputs is None:
        inputs = set(v for v in vertex_list if not parent_dict.get(v, []))
    elif isinstance(inputs, (set, list, tuple)):
        inputs = set(inputs)
    else:
        inputs = set(inputs)

    if outputs is None:
        children = {v: [] for v in vertex_list}
        for v in vertex_list:
            for p in parent_dict.get(v, []):
                children[p].append(v)
        outputs = set(v for v in vertex_list if not children[v])
    elif isinstance(outputs, (set, list, tuple)):
        outputs = set(outputs)
    else:
        outputs = set(outputs)

    # Build children map
    children = {v: [] for v in vertex_list}
    for v in vertex_list:
        for p in parent_dict.get(v, []):
            children[p].append(v)

    # Determine needed vertices
    needed = set(outputs)
    for v in reversed(vertex_list):
        if v in needed:
            for p in parent_dict.get(v, []):
                needed.add(p)

    needed_vertices = [v for v in vertex_list if v in needed]
    needed_set = set(needed_vertices)
    vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}

    # Precompute next-use positions
    next_use = defaultdict(list)
    for i, v in enumerate(needed_vertices):
        for p in parent_dict.get(v, []):
            if p in needed_set:
                next_use[p].append(i)

    # State
    in_fast = set()
    in_slow = set(inputs)
    computed = set()
    operations = []
    current_pos = 0

    def get_next_use(u, pos):
        uses = next_use.get(u, [])
        lo, hi = 0, len(uses)
        while lo < hi:
            mid = (lo + hi) // 2
            if uses[mid] <= pos:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(uses):
            return uses[lo]
        return float('inf')

    def evict_one(protect_set):
        if not in_fast:
            return None
        best_evict = None
        best_next_use = -1
        for u in list(in_fast):
            if u in protect_set:
                continue
            nu = get_next_use(u, current_pos)
            if nu > best_next_use:
                best_next_use = nu
                best_evict = u
        if best_evict is None:
            for u in in_fast:
                best_evict = u
                break
        if best_evict is not None:
            if best_evict in in_slow:
                operations.append(['delete', best_evict])
            else:
                operations.append(['store', best_evict])
            in_fast.discard(best_evict)
        return best_evict

    # Forward pass
    for v in needed_vertices:
        if time.time() > deadline:
            break
        current_pos = vertex_to_idx.get(v, current_pos)
        par_list = [p for p in parent_dict.get(v, []) if p in needed_set]
        
        for p in par_list:
            if p in in_fast:
                continue
            if len(in_fast) >= fast_memory_budget:
                remaining_parents = set(par_list) - {p}
                evict_one(remaining_parents | {v})
            operations.append(['load', p])
            in_fast.add(p)
        
        if v not in in_fast:
            operations.append(['compute', v])
            computed.add(v)
            in_fast.add(v)

    # Ensure outputs in fast
    for v in outputs:
        if v in needed_set and v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                evict_one(outputs)
            operations.append(['load', v])
            in_fast.add(v)

    for v in outputs:
        if v in needed_set and v not in in_fast:
            if len(in_fast) < fast_memory_budget:
                operations.append(['load', v])
                in_fast.add(v)
            else:
                if in_fast:
                    u = next(iter(in_fast))
                    if u in in_slow:
                        operations.append(['delete', u])
                    else:
                        operations.append(['store', u])
                    in_fast.discard(u)
                    operations.append(['load', v])
                    in_fast.add(v)

    return operations
# EVOLVE-BLOCK-END
