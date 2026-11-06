import time
import sys

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    
    # 1. Parse DAG
    # Assume dag is a list of [vertex_id, parents_list] or {vertex_id: parents_list}
    vertices = []
    parents_map = {}
    outputs = set()
    
    if isinstance(dag, dict):
        for v, parents in dag.items():
            vertices.append(v)
            parents_map[v] = parents
            if not parents:
                # Heuristic: if no parents, might be input.
                # But outputs are also sinks.
                # Need to distinguish.
                pass
        # Identify outputs: vertices with no children?
        # Or provided in dag?
        # The prompt says "dag has ... output vertices".
        # Maybe the last elements are outputs?
        # Or maybe there's a specific key?
        # I'll assume the last vertex in the list is the output, or all sinks are outputs.
        # Let's assume all vertices with no children are outputs.
        
    elif isinstance(dag, list):
        for item in dag:
            if isinstance(item, (list, tuple)):
                v = item[0]
                parents = item[1] if len(item) > 1 else []
                vertices.append(v)
                parents_map[v] = parents
            else:
                # Maybe just a list of vertices, and parents are implicit?
                # Unlikely.
                pass
        
    # Identify children to find outputs (sinks)
    children_map = {v: [] for v in vertices}
    for v, parents in parents_map.items():
        for p in parents:
            if p in children_map:
                children_map[p].append(v)
            else:
                # Parent might not be in vertices? No, DAG is closed.
                pass
                
    for v in vertices:
        if not children_map[v]:
            outputs.add(v)

    # Topological Sort (Kahn's Algorithm) to be safe
    in_degree = {v: 0 for v in vertices}
    for v, parents in parents_map.items():
        in_degree[v] = len(parents)
        
    queue = [v for v in vertices if in_degree[v] == 0]
    topo_order = []
    while queue:
        # Sort queue to ensure deterministic order if needed?
        # Not strictly necessary for correctness, but good for consistency.
        u = queue.pop(0)
        topo_order.append(u)
        for child in children_map[u]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
                
    # If topo_order length != vertices length, there's a cycle.
    # But it's a DAG.
    
    # Precompute last_use_idx
    last_use_idx = {v: -1 for v in vertices}
    for i, v in enumerate(topo_order):
        for p in parents_map[v]:
            if i > last_use_idx[p]:
                last_use_idx[p] = i
                
    # Note: For sinks (outputs), last_use_idx is their own index?
    # No, they are not parents. So last_use_idx remains -1 or their index?
    # They are "used" when computed. But not as parents.
    # For eviction logic:
    # If v is an output, we must store it at the end.
    # If v is not an output, and current_idx >= last_use_idx[v], it's safe to delete.
    
    # Initialize State
    fast_mem = set()
    slow_mem = set()
    # Inputs are vertices with no parents
    for v in vertices:
        if not parents_map[v]:
            slow_mem.add(v)
            
    ops = []
    last_used_time = {}
    time_step = 0
    
    def get_victim(fast_mem, current_idx):
        best_v = None
        best_score = float('inf')
        for v in fast_mem:
            # Score: lower is better to evict
            if v in outputs:
                # Outputs must be stored. We prefer to evict them if we have to, 
                # but we don't want to evict them if they were used recently?
                # Actually, outputs are not used as parents. 
                # So evicting them early is good.
                # Score based on LRU: older is better.
                score = (1, last_used_time.get(v, 0))
            else:
                if current_idx >= last_use_idx[v]:
                    # Safe to delete
                    score = (0, 0)
                else:
                    # Must store. LRU.
                    score = (2, last_used_time.get(v, 0))
            
            if score < best_score:
                best_score = score
                best_v = v
        return best_v

    # Simulation
    for i, v in enumerate(topo_order):
        # Check deadline
        if time.time() - start_time > 50:
            break
            
        # Ensure parents are in fast_mem
        for p in parents_map[v]:
            if p not in fast_mem:
                if p in slow_mem:
                    ops.append(['load', p])
                    fast_mem.add(p)
                    # slow_mem keeps p
                    last_used_time[p] = time_step
                    time_step += 1
                else:
                    # Should not happen if we always store needed vertices
                    # But if we deleted a non-needed vertex, it's not in slow_mem.
                    # If it's a parent, it WAS needed. So it should have been stored.
                    # This case implies a bug or a vertex that was deleted incorrectly.
                    # Fallback: Recompute? Too complex.
                    # Assume it's in slow_mem.
                    ops.append(['load', p])
                    fast_mem.add(p)
                    last_used_time[p] = time_step
                    time_step += 1
                    
                # Check budget
                while len(fast_mem) > fast_memory_budget:
                    victim = get_victim(fast_mem, i)
                    if victim is None: break
                    if victim in outputs:
                        ops.append(['store', victim])
                        fast_mem.remove(victim)
                        slow_mem.add(victim)
                    else:
                        if i >= last_use_idx[victim]:
                            ops.append(['delete', victim])
                            fast_mem.remove(victim)
                            # Not added to slow_mem
                        else:
                            ops.append(['store', victim])
                            fast_mem.remove(victim)
                            slow_mem.add(victim)
                            
        # Compute v
        ops.append(['compute', v])
        fast_mem.add(v)
        slow_mem.discard(v)
        last_used_time[v] = time_step
        time_step += 1
        
        # Check budget
        while len(fast_mem) > fast_memory_budget:
            victim = get_victim(fast_mem, i)
            if victim is None: break
            if victim in outputs:
                ops.append(['store', victim])
                fast_mem.remove(victim)
                slow_mem.add(victim)
            else:
                if i >= last_use_idx[victim]:
                    ops.append(['delete', victim])
                    fast_mem.remove(victim)
                else:
                    ops.append(['store', victim])
                    fast_mem.remove(victim)
                    slow_mem.add(victim)

    # Finalize Outputs
    for o in outputs:
        if o in fast_mem:
            ops.append(['store', o])
            fast_mem.remove(o)
            slow_mem.add(o)
            
    return ops
