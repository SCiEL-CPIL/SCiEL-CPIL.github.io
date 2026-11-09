import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50.0
    
    parents = dag.get('parents', {})
    outputs = set(dag.get('outputs', []))
    
    all_nodes = set(parents.keys())
    for ps in parents.values():
        for p in ps:
            all_nodes.add(p)
    all_nodes.update(outputs)
    
    # Topological Sort (Kahn's algorithm)
    in_degree = {v: len(parents.get(v, [])) for v in all_nodes}
    children = {v: [] for v in all_nodes}
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
            
    queue = [v for v in all_nodes if in_degree[v] == 0]
    topo_order = []
    while queue:
        v = queue.pop(0) # O(N) due to pop(0), use collections.deque
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    # If topo_order length != all_nodes, there's a cycle (shouldn't happen)
    
    in_slow = set(v for v in all_nodes if in_degree_orig[v] == 0) # Inputs
    # Wait, in_degree was modified. Recalculate inputs.
    inputs = [v for v in all_nodes if not parents.get(v, [])]
    in_slow = set(inputs)
    
    in_fast = set()
    computed = set()
    children_pending = {v: len(children[v]) for v in all_nodes}
    
    ops = []
    
    def evict():
        # Find a node in in_fast to evict
        # Prefer children_pending == 0
        # Then min children_pending
        candidates = list(in_fast)
        if not candidates:
            return None
        # Sort by children_pending, then arbitrary
        # To be fast, just find min
        min_pending = float('inf')
        best_v = None
        for v in candidates:
            if children_pending[v] < min_pending:
                min_pending = children_pending[v]
                best_v = v
        return best_v

    for v in topo_order:
        if time.time() > deadline:
            break
            
        if v in inputs:
            if v not in in_fast:
                if len(in_fast) >= fast_memory_budget:
                    u = evict()
                    if u is not None:
                        if children_pending[u] == 0:
                            in_fast.remove(u)
                            ops.append(['delete', u])
                        else:
                            in_fast.remove(u)
                            ops.append(['store', u])
                            in_slow.add(u)
                            ops.append(['delete', u])
                ops.append(['load', v])
                in_fast.add(v)
        else:
            # Ensure parents are in in_fast
            for p in parents.get(v, []):
                if p not in in_fast:
                    if p in in_slow:
                        if len(in_fast) >= fast_memory_budget:
                            u = evict()
                            if u is not None:
                                if children_pending[u] == 0:
                                    in_fast.remove(u)
                                    ops.append(['delete', u])
                                else:
                                    in_fast.remove(u)
                                    ops.append(['store', u])
                                    in_slow.add(u)
                                    ops.append(['delete', u])
                        ops.append(['load', p])
                        in_fast.add(p)
                    else:
                        # Should not happen
                        pass
                        
            # Compute v
            ops.append(['compute', v])
            in_fast.add(v)
            computed.add(v)
            
            # Update parents' pending
            for p in parents.get(v, []):
                children_pending[p] -= 1
                
            if v in outputs:
                ops.append(['store', v])
                in_slow.add(v)
                in_fast.remove(v)
                ops.append(['delete', v])
            else:
                # Check if we should evict v immediately?
                # No, we keep it unless we need space.
                # But if len(in_fast) > budget, we must evict someone.
                # We just added v, so len(in_fast) might be budget+1.
                if len(in_fast) > fast_memory_budget:
                    u = evict()
                    if u is not None:
                        if children_pending[u] == 0:
                            in_fast.remove(u)
                            ops.append(['delete', u])
                        else:
                            in_fast.remove(u)
                            ops.append(['store', u])
                            in_slow.add(u)
                            ops.append(['delete', u])
                            
    return ops
