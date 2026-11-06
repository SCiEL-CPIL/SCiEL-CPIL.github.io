import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin
    
    try:
        parents = dag.get('parents', {})
        outputs = set(dag.get('outputs', []))
        inputs = set(dag.get('inputs', []))
        
        # Build children
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
                
        all_vertices = set(parents.keys()) | inputs | outputs
        
        # Topological Order
        in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
        for v in inputs:
            in_degree[v] = 0
            
        queue = [v for v in all_vertices if in_degree.get(v, 0) == 0]
        topo_order = []
        while queue:
            queue.sort() # Deterministic
            u = queue.pop(0)
            topo_order.append(u)
            for c in children.get(u, []):
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
                    
        # Precompute depth (recompute cost proxy)
        depth = {}
        for v in topo_order:
            ps = parents.get(v, [])
            if not ps:
                depth[v] = 0
            else:
                depth[v] = max(depth[p] for p in ps) + 1
                
        # State: 'uncomputed', 'slow', 'fast'
        state = {v: 'uncomputed' for v in all_vertices}
        for v in inputs:
            state[v] = 'slow' # Inputs start blue (slow)
            
        fast_mem = set()
        ops = []
        
        # remaining_uses: number of children not yet computed
        remaining_uses = {v: len(children.get(v, [])) for v in all_vertices}
        
        def get_value(v):
            # Higher value = better to keep
            # If remaining_uses is 0, value is 0 (safe to evict)
            # Otherwise, value is proportional to uses and cost
            return remaining_uses[v] * (depth[v] + 1)
            
        def evict_one():
            # Find vertex in fast_mem with lowest value
            if not fast_mem:
                return
            best_v = None
            best_val = float('inf')
            for u in fast_mem:
                val = get_value(u)
                if val < best_val:
                    best_val = val
                    best_v = u
            if best_v is not None:
                ops.append(['store', best_v])
                state[best_v] = 'slow'
                fast_mem.discard(best_v)
                
        def ensure_fast(v):
            if state[v] == 'fast':
                return
            if state[v] == 'slow':
                # Need to load
                if len(fast_mem) >= fast_memory_budget:
                    evict_one()
                ops.append(['load', v])
                state[v] = 'fast'
                fast_mem.add(v)
            elif state[v] == 'uncomputed':
                # Need to compute (should not happen in topo order if we don't evict active nodes, 
                # but if we did evict an active node, it becomes 'slow', not 'uncomputed'.
                # 'uncomputed' only if we deleted it. We only store, so this shouldn't happen.)
                # But just in case:
                ps = parents.get(v, [])
                for p in ps:
                    ensure_fast(p)
                if len(fast_mem) >= fast_memory_budget:
                    evict_one()
                ops.append(['compute', v])
                state[v] = 'fast'
                fast_mem.add(v)
                # Update remaining_uses for parents? No, remaining_uses is for children.
                
        # Main Loop
        for i, v in enumerate(topo_order):
            if time.time() > DEADLINE:
                break
                
            # Ensure parents are fast
            ps = parents.get(v, [])
            for p in ps:
                ensure_fast(p)
                
            # Compute v
            # Check memory
            if len(fast_mem) >= fast_memory_budget:
                evict_one()
            ops.append(['compute', v])
            state[v] = 'fast'
            fast_mem.add(v)
            
            # Update remaining_uses for parents of v
            for p in ps:
                remaining_uses[p] -= 1
                # If remaining_uses[p] becomes 0, it's a candidate for eviction in future
                
            # If v is output, store it immediately to free space
            if v in outputs:
                ops.append(['store', v])
                state[v] = 'slow'
                fast_mem.discard(v)
                
            # Optional: Proactive eviction if memory is high and we have vertices with 0 uses?
            # The evict_one in the next iteration will handle it.
            
        # Final check: All outputs must be 'slow' (blue)
        # Our loop stores outputs immediately, so they are 'slow'.
        
        if len(ops) > 20000:
             # Truncate? Or return best effort.
             pass
             
        return ops
        
    except Exception:
        return []
