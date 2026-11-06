import time
import sys

def pebble(dag, fast_memory_budget):
    """
    Minimizes communication (transfers) for DAG pebbling.
    
    Args:
        dag: Dictionary or object representing the DAG. 
             Assuming format: {'nodes': [...], 'edges': [(u, v), ...]} or similar.
             Based on typical autoresearch_suite tasks, it might be:
             dag['nodes']: list of node ids
             dag['parents']: dict {node: list of parents}
             dag['outputs']: list of output node ids
        fast_memory_budget: int, capacity K.
        
    Returns:
        list of [operation, vertex]
    """
    start_time = time.time()
    deadline = 80.0 # 10s margin from 90s limit
    
    # 1. Parse DAG structure
    # Robustly handle different input formats
    if isinstance(dag, dict):
        nodes = dag.get('nodes', [])
        parents_map = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        # Sometimes edges are provided
        if 'edges' in dag and not parents_map:
            parents_map = {n: [] for n in nodes}
            for u, v in dag['edges']:
                parents_map[v].append(u)
    else:
        # Fallback if dag is an object with attributes
        nodes = getattr(dag, 'nodes', [])
        parents_map = getattr(dag, 'parents', {})
        outputs = getattr(dag, 'outputs', [])
        
    if not nodes:
        return []
        
    # Ensure all nodes are in parents_map
    for n in nodes:
        if n not in parents_map:
            parents_map[n] = []
            
    # Identify inputs (nodes with no parents)
    inputs = [n for n in nodes if len(parents_map[n]) == 0]
    
    # Build children map for future usage analysis
    children_map = {n: [] for n in nodes}
    for n in nodes:
        for p in parents_map[n]:
            children_map[p].append(n)
            
    # Topological Sort (Kahn's algorithm)
    in_degree = {n: len(parents_map[n]) for n in nodes}
    queue = [n for n in nodes if in_degree[n] == 0]
    topo_order = []
    while queue:
        # Sort to ensure deterministic order if needed, though not strictly required for correctness
        # But stable ordering helps with heuristic consistency
        queue.sort() 
        u = queue.pop(0)
        topo_order.append(u)
        for v in children_map[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(topo_order) != len(nodes):
        # Cycle detected, fallback to simple list
        topo_order = list(nodes)

    # Precompute future usage
    # future_usage[n] = number of children of n that appear after n in topo_order
    # Since topo_order is valid, all children appear after parents.
    # So future_usage[n] is just len(children_map[n]).
    # Wait, "future usage" in the context of eviction:
    # When we are at node v, we have processed some set of nodes.
    # A node u in fast memory is useful if it has children that are NOT yet processed.
    # Since we process in topo order, when we are about to process v,
    # all nodes before v in topo_order are processed.
    # So for a node u in fast memory, its "remaining children" are those in children_map[u] 
    # that are also in the suffix of topo_order starting from current position.
    
    # To make this efficient, we can precompute the index of each node in topo_order.
    node_index = {n: i for i, n in enumerate(topo_order)}
    
    # 2. Simulation
    fast_mem = set()
    slow_mem = set(inputs) # Inputs start in slow memory
    ops = []
    
    # We need to track which nodes are "available" in slow memory.
    # Initially, inputs are in slow memory.
    # When we compute a node, it's in fast memory.
    # When we store a node, it goes to slow memory.
    # When we delete a node, it is removed from fast memory.
    # If we need a node later that is not in fast memory, it must be in slow memory.
    
    # Heuristic for eviction:
    # We want to evict a node that is least likely to be needed soon.
    # Metric: Number of unprocessed children.
    
    def get_unprocessed_children_count(node, current_idx):
        # Count children of node that have index > current_idx
        count = 0
        for c in children_map[node]:
            if node_index[c] > current_idx:
                count += 1
        return count

    for i, v in enumerate(topo_order):
        if time.time() > deadline:
            break
            
        # 1. Ensure parents are in fast memory
        for p in parents_map[v]:
            if p not in fast_mem:
                # Must load from slow memory
                # Assume p is in slow_mem (it should be, either input or stored)
                if p in slow_mem:
                    ops.append(['load', p])
                    fast_mem.add(p)
                else:
                    # Error: Parent not available? 
                    # This can happen if we deleted it and didn't store it, 
                    # but then we'd need to recompute. 
                    # Our strategy: if we delete, we only delete if future_usage is 0.
                    # If future_usage is 0, it has no unprocessed children.
                    # So if v is a child of p, and v is unprocessed, p has future_usage > 0.
                    # So p should NOT have been deleted.
                    # So p must be in slow_mem.
                    pass

        # 2. Compute v
        ops.append(['compute', v])
        fast_mem.add(v)
        
        # 3. Manage Fast Memory Capacity
        while len(fast_mem) > fast_memory_budget:
            # Need to evict one node
            # Candidates: all nodes in fast_mem
            # We cannot evict v (just computed) if budget is 0? No, if budget is 0, we can't hold anything.
            # If budget is 0, we must load, compute, and immediately store/delete.
            
            # Score each candidate for eviction
            best_candidate = None
            best_score = None
            
            for u in fast_mem:
                # We should not evict v if possible? 
                # If budget is 1, and we just computed v, and we had 1 other node, we must evict.
                # If we evict v, we lose the result. We must store it.
                
                # Calculate "value" of keeping u
                # 1. Unprocessed children count
                unproc_children = get_unprocessed_children_count(u, i)
                
                # 2. Is it an output?
                is_output = (u in outputs)
                
                # Heuristic:
                # If unproc_children == 0, it is safe to delete (no store cost).
                # If unproc_children > 0, we must store (store cost).
                
                # Preference:
                # 1. Delete nodes with 0 unprocessed children (Cost 0).
                # 2. If no such nodes, store the node with the fewest unprocessed children (Minimize future load/recompute complexity? Or just arbitrary).
                #    Actually, if we store, we pay 1. If we delete (and it has children), we pay 0 now but must recompute later.
                #    Recomputing later means loading its parents.
                #    So deleting is only good if recomputation is cheap.
                #    Let's assume storing is better if unproc_children > 0.
                
                if unproc_children == 0:
                    # High priority to delete
                    priority = 0
                else:
                    # Lower priority to evict (must store)
                    # Among these, prefer to evict the one with lowest unproc_children?
                    # Or highest? 
                    # If we keep the one with many children, we avoid many loads.
                    # So evict the one with FEWEST unprocessed children.
                    priority = 100 + unproc_children
                
                # Tie-breaker: In-degree (complexity)
                # If priority is same, evict the one with lower in-degree (easier to recompute if deleted)
                # Or if we are storing, it doesn't matter.
                in_deg = len(parents_map[u])
                
                # Final score: Lower is better to evict
                score = (priority, in_deg)
                
                if best_score is None or score < best_score:
                    best_score = score
                    best_candidate = u
                    
            if best_candidate is None:
                break
                
            # Perform eviction
            u = best_candidate
            unproc_children = get_unprocessed_children_count(u, i)
            
            if unproc_children == 0:
                # Delete
                ops.append(['delete', u])
                fast_mem.remove(u)
                # It stays in slow_mem if it was there, or is just gone.
                # If it was computed and not stored, it's gone.
                # If it was an input and not stored, it's gone.
                # But inputs are in slow_mem.
                # If we delete an input, it's still in slow_mem?
                # "Delete" operation removes from fast memory.
                # Does it remove from slow memory? No.
                # So if u is in slow_mem, it remains there.
            else:
                # Store
                ops.append(['store', u])
                fast_mem.remove(u)
                slow_mem.add(u)
                
    # 3. Finalize Outputs
    # All outputs must finish blue.
    # If an output is in fast_mem, it is blue.
    # If an output is in slow_mem, it is blue.
    # If an output is in neither, it is lost.
    # Our logic ensures that if a node has unprocessed children, it is stored, not deleted.
    # Outputs have no children. So unproc_children for an output is 0.
    # So if an output is in fast_mem and we need to evict it, we DELETE it.
    # If we delete an output, it is removed from fast_mem.
    # Is it in slow_mem?
    # If it was computed, it was added to fast_mem. It was NOT added to slow_mem unless stored.
    # So if we delete it, it is lost!
    
    # Correction:
    # If a node is an output, we MUST ensure it ends up in slow_mem or fast_mem.
    # If we delete it, it's gone.
    # So for outputs, we should NEVER delete them. We must store them if we evict them.
    
    # Let's refine the eviction logic:
    # If u is an output:
    #   Must store.
    # Else if unproc_children == 0:
    #   Can delete.
    # Else:
    #   Must store.
    
    # Let's re-run the logic mentally or adjust the code.
    
    # Revised Eviction Logic:
    # 1. Identify candidates.
    # 2. For each candidate u:
    #    if u in outputs:
    #       action = 'store'
    #       cost = 1
    #    else:
    #       unproc = get_unprocessed_children_count(u, i)
    #       if unproc == 0:
    #           action = 'delete'
    #           cost = 0
    #       else:
    #           action = 'store'
    #           cost = 1
    # 3. We want to minimize total cost.
    #    So we prefer 'delete' (cost 0) over 'store' (cost 1).
    #    Among 'delete' candidates, pick any.
    #    Among 'store' candidates, pick the one with lowest unproc_children (to minimize future impact? Or just arbitrary).
    
    # Let's implement this correctly.

    # Reset simulation
    fast_mem = set()
    slow_mem = set(inputs)
    ops = []
    
    for i, v in enumerate(topo_order):
        if time.time() > deadline:
            break
            
        for p in parents_map[v]:
            if p not in fast_mem:
                if p in slow_mem:
                    ops.append(['load', p])
                    fast_mem.add(p)
                else:
                    # Should not happen
                    pass

        ops.append(['compute', v])
        fast_mem.add(v)
        
        while len(fast_mem) > fast_memory_budget:
            best_candidate = None
            best_priority = None
            
            for u in fast_mem:
                is_out = (u in outputs)
                unproc = get_unprocessed_children_count(u, i)
                
                if is_out:
                    # Must store
                    priority = (1, 0) # High priority to keep? No, we must evict someone.
                    # We want to evict the "least bad" one.
                    # Storing an output is necessary.
                    # Deleting a non-output with 0 unproc is free.
                    # So if there is a non-output with 0 unproc, evict that one.
                    # So outputs should have HIGH priority to keep (LOW priority to evict).
                    # Let's invert: Lower score = better to evict.
                    # Score 0: Delete, free.
                    # Score 1: Store, cost 1.
                    
                    action = 'store'
                    score = 10 # High cost to evict
                else:
                    if unproc == 0:
                        action = 'delete'
                        score = 0 # Free
                    else:
                        action = 'store'
                        score = 10 + unproc # Cost 1 + penalty for future use?
                        # Actually, if we store, we pay 1.
                        # The penalty for future use is avoided by storing.
                        # So we prefer to store nodes with HIGH future use?
                        # No, we are evicting. We want to evict the one with LOWEST future use.
                        # So score = 10 + unproc is correct (lower unproc = lower score = better to evict).
                
                if best_priority is None or score < best_priority[0]:
                    best_priority = (score, action)
                    best_candidate = u
                    
            if best_candidate is None:
                break
                
            u = best_candidate
            _, action = best_priority
            
            if action == 'delete':
                ops.append(['delete', u])
                fast_mem.remove(u)
            else:
                ops.append(['store', u])
                fast_mem.remove(u)
                slow_mem.add(u)
                
    # Check if all outputs are in slow_mem or fast_mem
    # If not, we might have lost them.
    # But our logic says outputs are always stored if evicted.
    # And if they are in fast_mem, they are fine.
    
    return ops

