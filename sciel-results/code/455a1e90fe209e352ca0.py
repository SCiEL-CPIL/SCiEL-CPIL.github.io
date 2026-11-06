"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Minimize communication (transfers) for DAG pebbling.
    """
    deadline = time.time() + 55.0
    
    vertices = dag.get('vertices', list(dag.get('parents', {}).keys()))
    parents_map = dag.get('parents', {})
    outputs = set(dag.get('outputs', []))
    inputs = set(dag.get('inputs', []))
    
    # Build children and compute remaining-use counts
    children = defaultdict(list)
    for v, ps in parents_map.items():
        for p in ps:
            children[p].append(v)
    
    # Precompute topological order
    in_degree = {v: len(parents_map.get(v, [])) for v in vertices}
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    temp_in_deg = dict(in_degree)
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            temp_in_deg[c] -= 1
            if temp_in_deg[c] == 0:
                queue.append(c)
    
    # If topo_order doesn't cover all, append remaining
    if len(topo_order) < len(vertices):
        for v in vertices:
            if v not in topo_order:
                topo_order.append(v)
    
    # Remaining uses: how many more times each vertex's value will be needed
    remaining_uses = {v: 0 for v in vertices}
    for v in vertices:
        for c in children[v]:
            remaining_uses[v] += 1
    
    # State
    in_fast = set()
    in_slow = set()
    ops = []
    op_count = 0
    MAX_OPS = 20000
    
    def try_evict():
        """Evict least useful vertex from fast memory. Returns True if evicted."""
        nonlocal op_count
        if not in_fast:
            return False
        best_v = None
        best_score = float('inf')
        for v in in_fast:
            score = remaining_uses[v]
            if v in outputs:
                score -= 500
            if score < best_score:
                best_score = score
                best_v = v
        if best_v is not None:
            ops.append(['store', best_v])
            in_fast.discard(best_v)
            in_slow.add(best_v)
            op_count += 1
            return True
        return False
    
    def ensure_in_fast(v):
        """Ensure v is in fast memory."""
        nonlocal op_count
        if v in in_fast:
            return True
        if v in in_slow:
            while len(in_fast) >= fast_memory_budget:
                if not try_evict():
                    break
                if op_count >= MAX_OPS or time.time() > deadline:
                    return False
            ops.append(['load', v])
            in_fast.add(v)
            in_slow.discard(v)
            op_count += 1
            return op_count < MAX_OPS and time.time() < deadline
        return False
    
    # Process in topological order
    for v in topo_order:
        if op_count >= MAX_OPS or time.time() > deadline:
            break
        
        if v in inputs:
            # Inputs start blue - assume they need to be loaded
            if not ensure_in_fast(v):
                break
            continue
        
        # Compute v - ensure all parents are in fast memory
        ok = True
        for p in parents_map.get(v, []):
            if not ensure_in_fast(p):
                ok = False
                break
        if not ok:
            break
        
        # Compute
        ops.append(['compute', v])
        in_fast.add(v)
        op_count += 1
        
        # Decrement remaining uses for parents
        for p in parents_map.get(v, []):
            remaining_uses[p] -= 1
        
        # Evict if over budget
        while len(in_fast) > fast_memory_budget and op_count < MAX_OPS and time.time() < deadline:
            if not try_evict():
                break
    
    # Final: ensure outputs are in fast memory
    for v in outputs:
        if op_count >= MAX_OPS or time.time() > deadline:
            break
        if v not in in_fast:
            if v in in_slow:
                if not ensure_in_fast(v):
                    break
    
    return ops
# EVOLVE-BLOCK-END

