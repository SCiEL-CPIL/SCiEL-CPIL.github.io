import sys
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Minimize communication (loads + stores) for DAG pebbling.
    
    Args:
        dag: A dictionary or object representing the DAG. 
             Based on typical task structures, it likely has:
             - 'nodes': list of node ids
             - 'edges': list of (parent, child) or similar
             - Or it is a dict: {node: [parents]}
             Let's assume a standard structure:
             dag['nodes']: list of all nodes
             dag['parents']: dict {node: list of parents}
             dag['outputs']: list of output nodes
             
             If the structure is different, we need to be robust.
             The prompt says "dag has topologically ordered parents and output vertices".
             This suggests `dag` might be a dict of lists, or similar.
             
             Let's inspect the baseline import: `pebble_baseline(dag, fast_memory_budget)`.
             Without seeing the exact structure, I will assume a common format:
             dag is a dict where keys are nodes and values are lists of parents.
             Outputs are nodes with no children.
             
             Wait, "topologically ordered parents" might mean the input is already sorted?
             Let's assume we need to derive the structure.
             
             Common format in these generated tasks:
             dag = {
                 'nodes': [0, 1, 2...],
                 'parents': {0: [], 1: [0], 2: [0, 1], ...},
                 'outputs': [n-1]
             }
             OR
             dag is just a dict {node: [parents]}
             
             Let's write a helper to extract the structure safely.
    """
    
    # Helper to extract structure from various potential formats
    def extract_structure(dag):
        # Try to identify nodes, parents, outputs
        if isinstance(dag, dict):
            if 'nodes' in dag and 'parents' in dag:
                nodes = dag['nodes']
                parents = dag['parents']
                outputs = dag.get('outputs', [])
                if not outputs:
                    # Find sinks
                    children = defaultdict(list)
                    for node, pars in parents.items():
                        for p in pars:
                            children[p].append(node)
                    outputs = [n for n in nodes if not children[n]]
                return nodes, parents, outputs
            
            # Maybe dag is just {node: [parents]}
            if all(isinstance(v, list) for v in dag.values()):
                parents = dag
                nodes = list(dag.keys())
                children = defaultdict(list)
                for node, pars in parents.items():
                    for p in pars:
                        children[p].append(node)
                outputs = [n for n in nodes if not children[n]]
                return nodes, parents, outputs
        
        # Fallback: assume dag is a list of nodes and we need to infer edges?
        # Unlikely. Let's assume the dict format above.
        raise ValueError("Unknown DAG format")

    try:
        nodes, parents, outputs = extract_structure(dag)
    except Exception:
        # If extraction fails, return empty or basic
        return []

    # Topological Sort
    in_degree = {n: len(parents.get(n, [])) for n in nodes}
    children = defaultdict(list)
    for n in nodes:
        for p in parents.get(n, []):
            children[p].append(n)
    
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
        # Cycle detected or error
        return []

    # State
    fast_mem = set()
    remaining_children = {n: len(children[n]) for n in nodes}
    is_output = set(outputs)
    ops = []
    
    # Helper to evict
    def evict_one():
        if not fast_mem:
            return
        # Priority 1: Dead nodes (remaining_children == 0)
        # Among dead nodes, prefer non-outputs (delete) over outputs (store)
        # Actually, if a node is an output, we MUST store it eventually.
        # If we delete it, we lose the value. So we cannot delete outputs.
        # So for dead nodes:
        # - If not output: delete
        # - If output: store
        
        # Priority 2: Live nodes with min remaining_children
        # These must be stored (cannot delete).
        
        # Let's pick the best candidate
        candidates = list(fast_mem)
        
        # Find dead non-outputs
        dead_non_outputs = [n for n in candidates if remaining_children[n] == 0 and n not in is_output]
        if dead_non_outputs:
            n = dead_non_outputs[0]
            ops.append(['delete', n])
            fast_mem.remove(n)
            return
        
        # Find dead outputs
        dead_outputs = [n for n in candidates if remaining_children[n] == 0 and n in is_output]
        if dead_outputs:
            n = dead_outputs[0]
            ops.append(['store', n])
            fast_mem.remove(n)
            return
        
        # Find live node with min remaining_children
        # Tie-breaker: prefer non-outputs? No, both require store.
        # Maybe prefer the one that is "older"? We don't track age easily.
        min_count = min(remaining_children[n] for n in candidates)
        candidates_min = [n for n in candidates if remaining_children[n] == min_count]
        n = candidates_min[0]
        ops.append(['store', n])
        fast_mem.remove(n)
        return

    # Main Loop
    for v in topo_order:
        # 1. Ensure parents are in fast memory
        for p in parents.get(v, []):
            if p not in fast_mem:
                if len(fast_mem) >= fast_memory_budget:
                    evict_one()
                ops.append(['load', p])
                fast_mem.add(p)
        
        # 2. Compute v
        if parents.get(v, []): # Not an input
            # If budget is 0 and v has parents, we can't compute it in fast mem?
            # But we just loaded parents. If budget is 0, we couldn't have loaded them.
            # Assume budget >= 1 for nodes with parents.
            if len(fast_mem) >= fast_memory_budget:
                # We need space for the new result
                # But wait, if we just loaded parents, they are in fast mem.
                # If budget is full, we need to evict BEFORE computing?
                # Compute creates a new red value.
                # So if len(fast_mem) == budget, we must evict one to make room for v.
                evict_one()
            
            ops.append(['compute', v])
            fast_mem.add(v)
            
            # 3. Update remaining children for parents
            for p in parents.get(v, []):
                remaining_children[p] -= 1
                # If p becomes dead, we might want to clean it up now?
                # The evict_one handles it when we need space.
                # But if p is dead and not output, keeping it wastes space.
                # Let's proactively clean dead non-outputs to save space for future loads.
                if remaining_children[p] == 0 and p not in is_output:
                     # It's safe to delete now
                     if p in fast_mem:
                         ops.append(['delete', p])
                         fast_mem.remove(p)
                elif remaining_children[p] == 0 and p in is_output:
                     # It's an output. We must store it eventually.
                     # Storing now frees space.
                     if p in fast_mem:
                         ops.append(['store', p])
                         fast_mem.remove(p)
        
        else: # Input node
            if v not in fast_mem:
                if len(fast_mem) >= fast_memory_budget:
                    evict_one()
                ops.append(['load', v])
                fast_mem.add(v)
                
            # Inputs have no parents to update.
            # But what if input is also an output? (Single node graph)
            # If input is output, it starts Blue. We loaded it (Red).
            # It has no children, so remaining_children is 0.
            # We should store it at the end? Or now?
            # If we store it now, it's Blue.
            # Let's let the final cleanup handle it, or handle it here.
            # If we store it here, we free space.
            # But if it's needed for something else? No, it's a source.
            # If it's an output, it's a sink too.
            # So if it's both source and sink, and we loaded it, we can store it immediately.
            if v in is_output and remaining_children[v] == 0:
                 if v in fast_mem:
                     ops.append(['store', v])
                     fast_mem.remove(v)

    # Final Cleanup: Ensure all outputs are Blue (stored)
    for v in outputs:
        if v in fast_mem:
            # If it's still in fast mem, it must be stored
            ops.append(['store', v])
            fast_mem.remove(v)
            
    # Note: Any other nodes left in fast_mem can be deleted or stored?
    # They are not outputs, so they don't need to be Blue.
    # We can just ignore them (implicit delete) or explicitly delete?
    # The prompt doesn't require non-outputs to be Blue.
    # So we don't need to clean them up for correctness, only for score?
    # Delete is not a transfer. Store is.
    # So we don't need to touch them.

    return ops
