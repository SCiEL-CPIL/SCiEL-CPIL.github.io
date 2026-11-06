import heapq
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    # Parse DAG
    # Assume dag is a list of dicts or tuples.
    # Common format: list of (node_id, parents, is_output)
    # Or list of dicts: {'id':..., 'parents':..., 'output':...}
    
    nodes = []
    parents_map = {}
    children_map = defaultdict(list)
    is_output = set()
    
    # Robust parsing
    for item in dag:
        if isinstance(item, dict):
            node_id = item.get('id') or item.get('node')
            pars = item.get('parents', [])
            out = item.get('output', False) or item.get('is_output', False)
        else:
            # Assume tuple/list: (id, parents, output)
            node_id = item[0]
            pars = item[1]
            out = item[2] if len(item) > 2 else False
            
        nodes.append(node_id)
        parents_map[node_id] = pars
        is_output.add(node_id) if out else None
        
        for p in pars:
            children_map[p].append(node_id)
            
    # Topological Sort (Kahn's algorithm) to be safe, or assume input is topo ordered
    # The prompt says "dag has topologically ordered parents", implying the input list might be in topo order.
    # But to be safe, I will perform a topo sort.
    
    in_degree = {n: len(parents_map[n]) for n in nodes}
    queue = [n for n in nodes if in_degree[n] == 0]
    topo_order = []
    
    # Use a copy of in_degree for sorting
    indeg = in_degree.copy()
    q = list(queue)
    while q:
        # To make it deterministic, sort queue? Or just pop.
        # For consistent results, sort by node id?
        q.sort()
        n = q.pop(0)
        topo_order.append(n)
        for child in children_map[n]:
            indeg[child] -= 1
            if indeg[child] == 0:
                q.append(child)
                
    if len(topo_order) != len(nodes):
        # Cycle? Should not happen in DAG.
        # Fallback to input order
        topo_order = nodes

    # State
    fm = set()
    remaining_children = {n: len(children_map[n]) for n in nodes}
    ops = []
    
    # Helper for eviction
    def evict_one():
        if not fm:
            return
        # Find node with min remaining_children
        # Tie-breaker: prefer non-output? Or just min rem_uses.
        # If rem_uses == 0, it's best.
        # Among rem_uses == 0, prefer delete (non-output) over store (output)?
        # Cost is same. But store puts it in slow. Delete removes it.
        # If it's an output, we MUST store it eventually. If we delete it, we lose it?
        # No, if it's an output and rem_uses==0, we store it.
        # If it's an intermediate and rem_uses==0, we delete it.
        
        # Selection:
        # 1. Min remaining_children
        # 2. If tie, prefer non-output (can be deleted, which is "free" space wise, same as store)
        #    Actually, store and delete are both 1 op.
        #    However, if we delete an intermediate, we don't pollute slow memory.
        #    If we store an output, we satisfy the end condition.
        
        best_node = None
        best_key = None
        
        for u in fm:
            rc = remaining_children[u]
            # We want to minimize rc.
            # Secondary: if rc == 0, we can delete if not output.
            # If rc > 0, we must store.
            
            # Key: (rc, is_output_flag)
            # If rc == 0:
            #   If not output: key (0, 0) -> Delete
            #   If output: key (0, 1) -> Store
            # If rc > 0:
            #   key (rc, 0) -> Store
            # 
            # Wait, if rc > 0, we MUST store.
            # If rc == 0 and not output, we DELETE.
            # If rc == 0 and output, we STORE.
            
            # We want to pick the one that is "most useless" for the future.
            # So min rc is primary.
            
            is_out = 1 if u in is_output else 0
            # We prefer to evict nodes with rc=0.
            # Among rc=0, we don't care much, but deleting is "cleaner".
            # However, the cost is identical.
            
            # Let's just pick min rc.
            key = rc
            
            if best_node is None or key < best_key:
                best_node = u
                best_key = key
                
        if best_node is not None:
            fm.remove(best_node)
            if remaining_children[best_node] == 0:
                if best_node in is_output:
                    ops.append(('store', best_node))
                else:
                    ops.append(('delete', best_node))
            else:
                ops.append(('store', best_node))

    # Main Loop
    for v in topo_order:
        # 1. Load Parents
        for p in parents_map[v]:
            if p not in fm:
                if len(fm) >= fast_memory_budget:
                    evict_one()
                ops.append(('load', p))
                fm.add(p)
                
        # 2. Compute v
        # If v is an input (no parents), we loaded it? No, inputs have no parents.
        # So for inputs, the "Load Parents" loop does nothing.
        # We must get v into FM.
        # If v has no parents, we `load` it.
        # If v has parents, we `compute` it.
        
        if len(parents_map[v]) == 0:
            # Input node
            if v not in fm:
                if len(fm) >= fast_memory_budget:
                    evict_one()
                ops.append(('load', v))
                fm.add(v)
        else:
            # Compute node
            # v is not in fm yet (unless it was loaded? No, we only load parents).
            # So v is not in fm.
            if len(fm) >= fast_memory_budget:
                evict_one()
            ops.append(('compute', v))
            fm.add(v)
            
        # 3. Update remaining_children for parents
        for p in parents_map[v]:
            remaining_children[p] -= 1

    # 4. Finalize
    for u in list(fm):
        if u in is_output:
            ops.append(('store', u))
        else:
            ops.append(('delete', u))
        fm.remove(u)
        
    return ops
