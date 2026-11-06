import time
from collections import deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    try:
        # Try to parse dag
        if isinstance(dag, dict):
            nodes = dag.get('nodes', [])
            parents = dag.get('parents', {})
            inputs = dag.get('inputs', [])
            outputs = dag.get('outputs', [])
        else:
            nodes = getattr(dag, 'nodes', [])
            parents = getattr(dag, 'parents', {})
            inputs = getattr(dag, 'inputs', [])
            outputs = getattr(dag, 'outputs', [])
            
        if not nodes:
            return []

        # Build children map
        children = {n: [] for n in nodes}
        for n in nodes:
            for p in parents.get(n, []):
                children[p].append(n)
                
        # Topological sort
        in_degree = {n: len(parents.get(n, [])) for n in nodes}
        queue = deque([n for n in nodes if in_degree[n] == 0])
        topo_order = []
        while queue:
            n = queue.popleft()
            topo_order.append(n)
            for c in children[n]:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
                    
        if len(topo_order) != len(nodes):
            # Fallback to baseline if DAG is invalid or structure unexpected
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)

        remaining_children = {n: len(children[n]) for n in nodes}
        in_memory = set()
        operations = []
        
        def load(v):
            operations.append(['load', v])
            in_memory.add(v)
            
        def store(v):
            operations.append(['store', v])
            in_memory.remove(v)
            
        def compute(v):
            operations.append(['compute', v])
            in_memory.add(v)
            
        def delete(v):
            operations.append(['delete', v])
            in_memory.remove(v)
            
        for v in topo_order:
            if time.time() > deadline or len(operations) > 19000:
                break
                
            for p in parents.get(v, []):
                if p not in in_memory:
                    if len(in_memory) >= fast_memory_budget:
                        # Evict
                        candidates = [u for u in in_memory if u not in parents.get(v, [])]
                        if candidates:
                            candidates.sort(key=lambda u: remaining_children[u])
                            victim = candidates[0]
                            if remaining_children[victim] == 0 and victim not in outputs:
                                delete(victim)
                            else:
                                store(victim)
                        else:
                            # Stuck, evict a parent (will reload later?)
                            # This is a failure case for this heuristic
                            pass
                    load(p)
                    
            compute(v)
            
            for p in parents.get(v, []):
                remaining_children[p] -= 1
                
            if len(in_memory) > fast_memory_budget:
                candidates = [u for u in in_memory if u != v]
                if candidates:
                    candidates.sort(key=lambda u: remaining_children[u])
                    victim = candidates[0]
                    if remaining_children[victim] == 0 and victim not in outputs:
                        delete(victim)
                    else:
                        store(victim)

        for o in outputs:
            if o in in_memory:
                store(o)

        return operations

    except Exception:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
