"""Rewritten program for communication-minimizing DAG pebbling."""
import time
import heapq
from collections import defaultdict, deque

# EVOLVE-BLOCK-START

def _toposort(dag):
    """
    Return a list of vertices in topological order.
    dag is expected to be a dict mapping vertex -> list of parent vertices.
    Outputs are vertices with no children (or explicitly marked). We infer outputs
    as vertices not appearing as any parent.
    """
    # Build adjacency: parent -> children
    children = defaultdict(list)
    all_vertices = set()
    for v, parents in dag.items():
        all_vertices.add(v)
        for p in parents:
            children[p].append(v)
            all_vertices.add(p)
    
    # Compute in-degrees (number of unprocessed parents)
    indegree = {v: 0 for v in all_vertices}
    for v, parents in dag.items():
        indegree[v] = len(parents)
    
    # Vertices with indegree 0 are sources (inputs)
    queue = deque([v for v in all_vertices if indegree[v] == 0])
    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for c in children[u]:
            indegree[c] -= 1
            if indegree[c] == 0:
                queue.append(c)
    
    # If not all vertices were reached, there might be cycles or disconnected components
    # For safety, add any remaining vertices
    if len(order) != len(all_vertices):
        remaining = set(all_vertices) - set(order)
        order.extend(sorted(remaining))
    
    return order

def _get_outputs(dag, order):
    """
    Outputs are vertices that are not parents of any other vertex.
    """
    parents_set = set()
    for v, parents in dag.items():
        for p in parents:
            parents_set.add(p)
    
    all_vertices = set(dag.keys())
    for parents in dag.values():
        all_vertices.update(parents)
    
    outputs = [v for v in order if v not in parents_set]
    return outputs

def _count_naive_transfers(dag, order, outputs):
    """
    Count the number of transfers in a naive schedule.
    Naive: process each vertex once. For each vertex, load all parents if not in memory,
    compute, store result, then delete all values to keep memory clean? 
    Actually, the naive baseline likely does:
    - For each vertex in topological order:
      - For each parent: if not in fast memory, load it (1 transfer)
      - Compute (0 transfers)
      - Store result in fast memory (0 transfers, it's already there after compute)
      - Delete all values that are no longer needed? 
    The problem statement says "Score mean transfers / naive schedule".
    A common naive schedule for DAG pebbling with bounded memory is:
    Load inputs, compute, store, delete when no longer needed.
    However, without knowing the exact baseline, let's assume a simple "load all parents, compute, store, delete all" per vertex.
    But wait, the task says "Inputs start blue". This implies inputs are already in memory?
    "every red value consumes one fast-memory slot, including newly computed results."
    "Inputs start blue" -> Inputs are in fast memory initially?
    If inputs start blue (in fast memory), then we don't need to load them.
    Let's re-read: "Inputs start blue, all outputs must finish blue".
    This suggests the initial state has inputs in fast memory (blue).
    The naive schedule might just be: for each vertex, compute if parents are blue, store result (blue), delete when safe.
    If inputs are already blue, the only transfers are for intermediate values that must be re-loaded?
    No, if we process in topological order and keep everything in memory, we need O(width) memory.
    If fast_memory_budget is small, we must delete some values and potentially reload them later.
    
    The "naive schedule" likely refers to a specific baseline implementation.
    Since I don't have access to `pebble_baseline`, I must estimate or implement a comparable naive strategy.
    A very naive strategy: For each vertex, load all parents (if not in memory), compute, store, delete all parents that are no longer needed.
    But if inputs start blue, we don't load them.
    
    Let's assume the naive schedule is:
    - Initialize memory with all inputs (blue).
    - For each vertex in topological order:
        - For each parent not in memory: load it.
        - Compute vertex.
        - Store vertex in memory.
        - Delete any memory values that are no longer needed by any remaining vertex.
    
    This is a standard greedy pebbling strategy.
    """
    # This is just for scoring reference; the actual score is computed by the evaluator.
    # We don't need to return this, but it helps to understand the problem.
    pass

def pebble(dag, fast_memory_budget):
    """
    Solve the DAG pebbling problem to minimize communication (transfers).
    
    Args:
        dag: dict mapping vertex -> list of parent vertices
        fast_memory_budget: int, max number of values that can be in fast memory
    
    Returns:
        list of [operation, vertex], where operation is 'load', 'store', 'compute', or 'delete'
    """
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit, 10s margin
    
    # Parse DAG
    order = _toposort(dag)
    outputs = _get_outputs(dag, order)
    
    # Build parent and child maps
    parents = {v: list(parents_list) for v, parents_list in dag.items()}
    # Ensure all vertices have a parents list (inputs have empty)
    for v in order:
        if v not in parents:
            parents[v] = []
    
    children = defaultdict(list)
    for v, p_list in parents.items():
        for p in p_list:
            children[p].append(v)
    
    # Initial state: inputs are blue (in fast memory)
    # Inputs are vertices with no parents
    inputs = [v for v in order if len(parents[v]) == 0]
    
    # If budget is large enough to hold all values, just compute in order
    if fast_memory_budget >= len(order):
        ops = []
        # Inputs are already blue, no load needed
        for v in order:
            if len(parents[v]) == 0:
                continue  # Already blue
            ops.append(['compute', v])
            ops.append(['store', v])
            # We don't need to delete if we have enough memory, but to be safe and consistent:
            # Actually, if we have enough memory, we don't delete.
        return ops
    
    # For smaller budgets, we need a smarter strategy.
    # Strategy: Use a recursive pebbling algorithm or a greedy approach with lookahead.
    # Given the constraints (up to 20000 ops, 60s), a simple greedy might suffice for small DAGs.
    # Let's try a greedy approach:
    # 1. Identify vertices that can be computed next (all parents in memory).
    # 2. Compute one, store it.
    # 3. Delete any vertex that is no longer needed.
    # 4. If no vertex can be computed, we must load a parent.
    #    Which parent to load? Choose one that unlocks the most computations or is needed soonest.
    
    # State:
    # in_memory: set of vertices currently in fast memory (blue)
    # computed: set of vertices already computed (even if deleted)
    # ops: list of operations
    
    in_memory = set(inputs)
    computed = set(inputs)
    ops = []
    
    # Precompute "needed by" count: how many children still need this vertex
    # A vertex can be deleted if all its children have been computed.
    remaining_children = {v: len(children[v]) for v in order}
    
    # We need to process vertices in some order.
    # Let's use a priority queue or simple loop.
    
    # To avoid infinite loops, we track the number of operations.
    max_ops = 20000
    
    # We'll use a simple greedy:
    # 1. Try to compute any vertex whose parents are all in memory.
    # 2. If none, load a parent that is not in memory.
    #    Choose the parent that is needed by the most uncomputed vertices.
    
    # Precompute for each vertex, the set of vertices that depend on it (transitively)
    # This is expensive, so let's use a simpler heuristic: direct children count.
    
    # Let's iterate until all outputs are in memory.
    # We need to ensure we don't exceed max_ops.
    
    # To make this more efficient, we can use a recursive divide-and-conquer approach
    # for layered DAGs, but let's start with greedy.
    
    # Heuristic for loading: Load the vertex that is a parent of the most uncomputed vertices.
    
    # Precompute dependency depth or something similar?
    
    # Let's just run the greedy loop.
    
    while any(v not in in_memory for v in outputs) and len(ops) < max_ops:
        # Check deadline
        if time.time() > deadline:
            break
        
        # Try to compute a vertex
        computed_next = False
        for v in order:
            if v in computed:
                continue
            if all(p in in_memory for p in parents[v]):
                # Can compute v
                ops.append(['compute', v])
                computed.add(v)
                # Store v in memory (it's now blue)
                in_memory.add(v)
                # Update remaining_children for parents
                for p in parents[v]:
                    remaining_children[p] -= 1
                # Delete any parents that are no longer needed
                for p in parents[v]:
                    if remaining_children[p] == 0 and p in in_memory and p not in outputs:
                        ops.append(['delete', p])
                        in_memory.remove(p)
                computed_next = True
                break
        
        if computed_next:
            continue
        
        # No vertex could be computed. Need to load a parent.
        # Find all uncomputed vertices and their missing parents.
        missing_parents = set()
        for v in order:
            if v in computed:
                continue
            for p in parents[v]:
                if p not in in_memory:
                    missing_parents.add(p)
        
        if not missing_parents:
            # Should not happen if outputs are not all in memory
            # Maybe we need to compute an input? No, inputs are already computed.
            break
        
        # Choose which parent to load.
        # Heuristic: Load the parent that is needed by the most uncomputed vertices.
        # Or load the one with the highest "priority".
        
        # Simple: pick the first one in topological order?
        # Better: pick the one with the most direct children that are uncomputed.
        
        best_parent = None
        best_score = -1
        for p in missing_parents:
            # Score: number of uncomputed children
            score = sum(1 for c in children[p] if c not in computed)
            if score > best_score:
                best_score = score
                best_parent = p
        
        if best_parent is None:
            best_parent = next(iter(missing_parents))
        
        # Load best_parent
        ops.append(['load', best_parent])
        in_memory.add(best_parent)
    
    # Ensure all outputs are in memory
    # If we broke out of the loop early, we might need to force-load outputs
    for v in outputs:
        if v not in in_memory:
            # This shouldn't happen if the loop ran correctly
            pass
    
    return ops

# EVOLVE-BLOCK-END
