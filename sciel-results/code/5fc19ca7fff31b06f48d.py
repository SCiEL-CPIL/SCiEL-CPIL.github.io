"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from collections import defaultdict, deque

def _topo_sort(vertices, parents_map):
    """Topologically sort vertices given parents_map (vertex -> list of parents)."""
    indeg = {v: 0 for v in vertices}
    children = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children[p].append(v)
            indeg[v] += 1
    q = deque([v for v in vertices if indeg[v] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for w in children[u]:
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    return order

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    dag: dict with keys:
      - 'vertices': list of vertex IDs
      - 'parents': dict mapping vertex -> list of parent vertex IDs (topologically ordered)
      - 'outputs': list of output vertex IDs
      - 'inputs': list of input vertex IDs (start blue)
    fast_memory_budget: int, number of fast-memory slots
    
    Returns: list of [operation, vertex] where operation is one of 'load', 'store', 'compute', 'delete'
    """
    start_time = time.time()
    deadline = start_time + 50  # safety margin
    
    vertices = dag['vertices']
    parents_map = dag.get('parents', {})
    outputs = set(dag.get('outputs', []))
    inputs = set(dag.get('inputs', []))
    
    # Ensure all vertices are known
    all_vertices = set(vertices)
    for v, ps in parents_map.items():
        for p in ps:
            all_vertices.add(p)
    vertices = list(all_vertices)
    
    # Topological order
    topo = _topo_sort(vertices, parents_map)
    topo_set = set(topo)
    
    # Build children map
    children = defaultdict(list)
    for v in topo:
        for p in parents_map.get(v, []):
            children[p].append(v)
    
    # Determine which vertices are inputs (no parents or explicitly inputs)
    input_set = inputs if inputs else set(v for v in topo if not parents_map.get(v, []))
    
    # Count number of times each vertex is needed (fan-out in reverse)
    # A vertex is "needed" if it's an output or has children that need it
    needed_count = defaultdict(int)
    for v in reversed(topo):
        if v in outputs:
            needed_count[v] = 1
        for c in children[v]:
            if needed_count[c] > 0:
                needed_count[v] += needed_count[c]
        # Cap to avoid overflow
        if needed_count[v] > 10000:
            needed_count[v] = 10000
    
    # Strategy: Greedy with look-ahead
    # We'll use a simple but effective heuristic:
    # 1. Process vertices in topological order
    # 2. Keep values in fast memory when possible
    # 3. Re-load from slow memory when necessary
    
    # Track state:
    # fast_mem: set of vertices currently in fast memory (red)
    # slow_mem: dict of vertex -> value (all computed values are in slow memory)
    # blue: set of vertices that are "blue" (available without loading)
    
    # Actually, let's think about this differently.
    # The pebbling model:
    # - Blue pebble: vertex is in fast memory (or is an input)
    # - Red pebble: vertex is stored in slow memory
    # - No pebble: vertex value is not available
    
    # Operations:
    # - load v: move v from slow to fast (cost: 1 transfer)
    # - store v: move v from fast to slow (cost: 1 transfer)
    # - compute v: compute v from its parents (all parents must be blue/fast)
    # - delete v: remove v from fast memory (no transfer cost)
    
    # Goal: All outputs must be blue (in fast memory) at the end.
    # Minimize total transfers (load + store operations).
    
    # Simple baseline: topological order with minimal memory management
    # For each vertex in topo order:
    #   - If it's an input, it's already blue
    #   - Otherwise, ensure all parents are blue, compute it
    #   - If memory is full, store some vertices to make room
    #   - At the end, ensure all outputs are blue
    
    # Let's implement a greedy strategy with memory management
    
    # State:
    fast_mem = set()  # vertices in fast memory (blue)
    slow_mem = set()  # vertices stored in slow memory (red)
    
    # Inputs start blue
    for v in input_set:
        fast_mem.add(v)
    
    ops = []
    
    def is_computable(v):
        """Check if all parents of v are in fast memory."""
        for p in parents_map.get(v, []):
            if p not in fast_mem:
                return False
        return True
    
    def find_vertex_to_store():
        """Find a vertex in fast memory to store to make room.
        Prefer vertices that are not needed soon or have low re-use value."""
        candidates = []
        for v in fast_mem:
            if v in input_set:
                continue  # Don't store inputs? Actually we can, but they're cheap to reload
            if v in outputs:
                continue  # Don't store outputs until the end
            # Score: lower is better to store
            # Prefer vertices with lower needed_count (less re-use)
            score = needed_count.get(v, 0)
            candidates.append((score, v))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]
    
    # Process in topological order
    for v in topo:
        if time.time() > deadline:
            break
        
        if v in input_set:
            # Already in fast memory
            continue
        
        # Check if v is already computed and in fast memory
        if v in fast_mem:
            continue
        
        # Need to compute v
        # First, ensure all parents are in fast memory
        for p in parents_map.get(v, []):
            if p not in fast_mem:
                if p in slow_mem:
                    # Load from slow memory
                    # Check if we need to make room
                    if len(fast_mem) >= fast_memory_budget:
                        # Store a vertex to make room
                        victim = find_vertex_to_store()
                        if victim is not None:
                            ops.append(['store', victim])
                            fast_mem.discard(victim)
                            slow_mem.add(victim)
                    ops.append(['load', p])
                    fast_mem.add(p)
                    slow_mem.discard(p)
                else:
                    # Parent not in slow memory either - shouldn't happen in valid DAG
                    # But handle gracefully: recompute if possible
                    # For now, just add a load (will be handled by verifier)
                    if len(fast_mem) >= fast_memory_budget:
                        victim = find_vertex_to_store()
                        if victim is not None:
                            ops.append(['store', victim])
                            fast_mem.discard(victim)
                            slow_mem.add(victim)
                    ops.append(['load', p])
                    fast_mem.add(p)
        
        # Now compute v
        if len(fast_mem) >= fast_memory_budget:
            # Need to make room for the new computed value
            victim = find_vertex_to_store()
            if victim is not None:
                ops.append(['store', victim])
                fast_mem.discard(victim)
                slow_mem.add(victim)
        
        ops.append(['compute', v])
        fast_mem.add(v)
        slow_mem.add(v)  # Also store in slow memory for potential reload
    
    # Ensure all outputs are in fast memory
    for v in outputs:
        if time.time() > deadline:
            break
        if v not in fast_mem:
            if v in slow_mem:
                if len(fast_mem) >= fast_memory_budget:
                    victim = find_vertex_to_store()
                    if victim is not None:
                        ops.append(['store', victim])
                        fast_mem.discard(victim)
                        slow_mem.add(victim)
                ops.append(['load', v])
                fast_mem.add(v)
                slow_mem.discard(v)
            else:
                # Need to recompute - complex case, skip for now
                pass
    
    # Truncate to 20000 operations if needed
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
# EVOLVE-BLOCK-END
