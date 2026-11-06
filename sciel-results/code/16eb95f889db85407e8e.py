def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Strategy:
    1. Process nodes in topological order.
    2. Load parents into fast memory if not already present.
    3. Compute the node.
    4. Update the 'remaining uses' count for parents.
    5. If a parent's remaining uses drop to 0, delete it from fast memory to free space.
    6. If the node is an output, store it to slow memory.
    7. If the node is an intermediate and fast memory is over budget, evict the node 
       with the fewest remaining uses.
    """
    import sys

    # --- Parse the DAG structure ---
    # We need to extract:
    # - Topological order of nodes
    # - Parents for each node
    # - Set of output nodes
    
    nodes = []
    parents = {}
    outputs = set()

    # Heuristic parsing for common DAG representations
    if isinstance(dag, list):
        # Case 1: List of node dictionaries
        # e.g., [{'id': 0, 'parents': [], 'is_output': False}, ...]
        for i, node in enumerate(dag):
            if isinstance(node, dict):
                nid = node.get('id', i)
                p_list = node.get('parents', [])
                is_out = node.get('is_output', node.get('output', False))
                
                nodes.append(nid)
                parents[nid] = p_list
                if is_out:
                    outputs.add(nid)
            else:
                # Case 2: List of tuples (node_id, parents_list)? Unlikely.
                # Or just IDs? No, need parents.
                pass

    elif isinstance(dag, dict):
        # Case 3: Dictionary with 'nodes', 'parents', 'outputs'
        if 'nodes' in dag:
            node_list = dag['nodes']
            if isinstance(node_list, list):
                for i, node in enumerate(node_list):
                    if isinstance(node, dict):
                        nid = node.get('id', i)
                        p_list = node.get('parents', [])
                        is_out = node.get('is_output', node.get('output', False))
                        nodes.append(nid)
                        parents[nid] = p_list
                        if is_out:
                            outputs.add(nid)
                    else:
                        nodes.append(node)
                        parents[node] = []
                        # Check for outputs in a separate list
                if 'outputs' in dag:
                    outputs = set(dag['outputs'])
                if 'parents' in dag and isinstance(dag['parents'], dict):
                    parents = dag['parents']
            elif isinstance(node_list, dict):
                # Nodes as dict: {id: {parents: [...]}}
                for nid, info in node_list.items():
                    nodes.append(nid)
                    if isinstance(info, dict):
                        parents[nid] = info.get('parents', [])
                    else:
                        parents[nid] = []
                if 'outputs' in dag:
                    outputs = set(dag['outputs'])
                
        # Case 4: The dict itself is the parent map? {node: [parents]}
        # And outputs are determined by having no children?
        # This is ambiguous. We'll stick to the explicit 'outputs' key if present.
        # If no 'nodes' key, assume the dict is {node_id: [parents]}
        if 'nodes' not in dag:
            # Assume dag is {node: [parents]}
            # We need to determine topological order.
            # If the dict keys are already topologically sorted, we can use that.
            # Python dicts preserve insertion order (3.7+).
            for nid, p_list in dag.items():
                nodes.append(nid)
                parents[nid] = p_list
            # Identify outputs: nodes that are not parents of any other node
            all_children = set()
            for p_list in parents.values():
                all_children.update(p_list)
            # Nodes not in all_children are potential outputs (sinks)
            # But the task says "output vertices" are specified.
            # If not specified, we assume all sinks are outputs?
            # Or maybe there's a separate key?
            # Let's check for 'outputs' key again.
            if 'outputs' in dag:
                outputs = set(dag['outputs'])
            else:
                # Fallback: Sinks are outputs
                outputs = set(nodes) - all_children

    # Fallback if parsing failed (empty nodes)
    if not nodes:
        return []

    # --- Pebbling Logic ---
    
    # Initialize state
    fast_mem = set()
    ops = []
    
    # Calculate initial 'uses_left' for each node
    # uses_left[v] = number of children v has
    children_count = {v: 0 for v in nodes}
    for v in nodes:
        for p in parents.get(v, []):
            if p in children_count:
                children_count[p] += 1
    
    uses_left = children_count.copy()

    # Process nodes in topological order
    for v in nodes:
        # 1. Load parents into fast memory
        v_parents = parents.get(v, [])
        for p in v_parents:
            if p not in fast_mem:
                ops.append(['load', p])
                fast_mem.add(p)
        
        # 2. Compute v
        ops.append(['compute', v])
        fast_mem.add(v)
        
        # 3. Update uses_left for parents
        # Since we just processed v, v no longer needs its parents.
        for p in v_parents:
            if p in uses_left:
                uses_left[p] -= 1
                # If a parent is no longer needed, delete it to free space
                if uses_left[p] == 0 and p in fast_mem:
                    ops.append(['delete', p])
                    fast_mem.remove(p)
        
        # 4. Handle Output vs Intermediate
        if v in outputs:
            # Outputs must be stored to slow memory
            ops.append(['store', v])
            fast_mem.remove(v)
            # uses_left[v] is no longer relevant
        else:
            # Intermediate node
            # Check if we are over budget
            while len(fast_mem) > fast_memory_budget:
                # Find the node in fast_mem with the minimum uses_left
                # If all have uses_left > 0, we pick the one with the smallest count.
                # If some have uses_left == 0, they should have been deleted in step 3.
                # But v itself might have uses_left > 0.
                
                # Optimization: If v has the lowest uses_left, delete v?
                # Yes, min handles that.
                
                # Note: It's possible that fast_memory_budget is 0?
                # If budget is 0, we can't hold anything.
                # But we just added v. So len(fast_mem) >= 1.
                # If budget == 0, we must delete something.
                
                victim = min(fast_mem, key=lambda x: uses_left.get(x, 0))
                ops.append(['delete', victim])
                fast_mem.remove(victim)
                
                # If we deleted v itself, that's fine.
                # If we deleted a parent, that's fine.

    return ops
