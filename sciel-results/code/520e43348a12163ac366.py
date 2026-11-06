"""Communication-minimizing DAG pebbling solver."""
import heapq
from collections import deque, defaultdict

# EVOLVE-BLOCK-START

def pebble(dag, fast_memory_budget):
    """
    Pebble a DAG with minimum total transfers (load+store+delete), subject to
    fast_memory_budget constraint on number of simultaneously red values.
    
    Parameters:
        dag: dict with keys:
            - 'parents': dict mapping vertex -> list of parent vertices
            - 'vertices': list of all vertex IDs
            - 'inputs': list of input vertex IDs
            - 'outputs': list of output vertex IDs
        fast_memory_budget: int, max number of red (in fast memory) vertices at once
    
    Returns:
        list of [operation, vertex] where operation in {'load','store','compute','delete'}
    """
    parents = dag['parents']
    vertices = list(dag['vertices'])
    inputs = set(dag['inputs'])
    outputs = set(dag['outputs'])
    
    # Build adjacency: children mapping
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Topological order (assumed parents are topologically ordered, but verify)
    # Kahn's algorithm
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    topo_order = []
    queue = deque([v for v in vertices if in_degree[v] == 0])
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(vertices):
        # Fallback: use given order
        topo_order = vertices
    
    # Precompute for each vertex: set of descendants (vertices that depend on it)
    # Actually we need: for each vertex v, the set of vertices that have v as an ancestor
    # Let's compute "future_users": for each vertex v, how many times it's needed in future
    # More precisely: for each vertex v, the set of vertices in topo_order that come after v
    # and have v as an ancestor.
    
    # Compute ancestor sets efficiently
    # For each vertex, compute the set of its descendants
    descendants = {v: set() for v in vertices}
    for v in reversed(topo_order):
        for c in children[v]:
            descendants[v].update(descendants[c])
            descendants[v].add(c)
    
    # For eviction heuristic: when we need to evict, prefer vertices whose descendants
    # are all already processed (no future use), or have minimal future use
    
    # State: (cost, num_ops, frozenset of red vertices, next_index, ops_list)
    # But state space is too large for Dijkstra on subsets.
    
    # Instead, use a greedy approach with smart eviction:
    # Process vertices in topological order.
    # Maintain set of vertices currently in fast memory (red).
    # For each vertex v in topo order:
    #   1. Ensure all parents of v are in fast memory (load if not)
    #   2. If fast memory is full and v is not in fast memory, evict a vertex
    #   3. Compute v (add to fast memory)
    #   4. If v is not an output and not needed in future, delete it
    
    # Better: use a priority-based eviction
    # Track for each vertex in fast memory: the last index in topo_order where it's needed
    # When we need to evict, evict the one with the earliest "last needed" time
    
    # Precompute for each vertex v: the last position in topo_order where v is used as a parent
    # i.e., the maximum index i such that there exists a vertex u at topo_order[i] with v in parents[u]
    
    last_needed = {}
    for i, v in enumerate(topo_order):
        for p in parents.get(v, []):
            last_needed[p] = i  # p is needed at position i
    
    # For input vertices, they may be needed at multiple positions
    # Actually, let's track all positions where a vertex is needed
    needed_at = defaultdict(list)
    for i, v in enumerate(topo_order):
        for p in parents.get(v, []):
            needed_at[p].append(i)
    
    # For each vertex, the last time it's needed
    for v in needed_at:
        last_needed[v] = max(needed_at[v])
    
    # Now process
    in_fast = set()
    ops = []
    op_count = 0
    
    # Input vertices start blue, so we need to load them when needed
    
    for i, v in enumerate(topo_order):
        # Step 1: Ensure all parents are in fast memory
        for p in parents.get(v, []):
            if p not in in_fast:
                # Need to load p
                if len(in_fast) >= fast_memory_budget:
                    # Need to evict someone
                    # Evict the vertex with the earliest last_needed that is not needed for current or future
                    # Candidates: vertices in in_fast that are not parents of v and not needed at position i or later
                    candidates = []
                    for u in in_fast:
                        # u is not a parent of v
                        if u == v:
                            continue
                        if u in parents.get(v, []):
                            continue
                        # Check if u is needed at position i or later
                        u_last = last_needed.get(u, -1)
                        if u_last < i:
                            # u is not needed anymore
                            candidates.append((0, u))
                        else:
                            # u is needed in future, higher cost to evict
                            candidates.append((u_last, u))
                    
                    if not candidates:
                        # All in fast are needed, pick one with smallest last_needed
                        candidates = [(last_needed.get(u, 0), u) for u in in_fast if u != v]
                        if not candidates:
                            candidates = [(0, u) for u in in_fast]
                    
                    # Evict the one with smallest last_needed
                    candidates.sort()
                    victim = candidates[0][1]
                    in_fast.remove(victim)
                    ops.append(['delete', victim])
                    op_count += 1
                
                in_fast.add(p)
                ops.append(['load', p])
                op_count += 1
        
        # Step 2: Compute v
        if len(in_fast) >= fast_memory_budget and v not in in_fast:
            # Need to evict someone to make room for v
            candidates = []
            for u in in_fast:
                if u in parents.get(v, []):
                    continue
                u_last = last_needed.get(u, -1)
                candidates.append((u_last, u))
            
            if not candidates:
                candidates = [(0, u) for u in in_fast if u != v]
                if not candidates:
                    candidates = [(0, u) for u in in_fast]
            
            candidates.sort()
            victim = candidates[0][1]
            in_fast.remove(victim)
            ops.append(['delete', victim])
            op_count += 1
        
        if v not in in_fast:
            in_fast.add(v)
        ops.append(['compute', v])
        op_count += 1
        
        # Step 3: If v is not an output and not needed in future, delete it
        if v not in outputs:
            v_last = last_needed.get(v, -1)
            if v_last < i + 1:  # v is not needed after this position
                if v in in_fast:
                    in_fast.remove(v)
                    ops.append(['delete', v])
                    op_count += 1
    
    # At the end, all outputs must be in fast memory (blue at end means they're stored)
    # Actually, the problem says "all outputs must finish blue" - meaning they need to be in fast memory?
    # Wait: "Inputs start blue, all outputs must finish blue" - blue means in fast memory
    # So we need to ensure outputs are in fast memory at the end.
    
    # Check if all outputs are in fast memory
    for o in outputs:
        if o not in in_fast:
            # Load it
            if len(in_fast) >= fast_memory_budget:
                candidates = [(last_needed.get(u, 0), u) for u in in_fast if u != o]
                candidates.sort()
                if candidates:
                    victim = candidates[0][1]
                    in_fast.remove(victim)
                    ops.append(['delete', victim])
                    op_count += 1
            in_fast.add(o)
            ops.append(['load', o])
            op_count += 1
    
    # If we exceeded 20000 ops, truncate (shouldn't happen for reasonable inputs)
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops

# EVOLVE-BLOCK-END
