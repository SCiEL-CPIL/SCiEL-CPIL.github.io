"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def pebble(dag, fast_memory_budget):
    """
    Minimize communication for DAG pebbling.
    
    Args:
        dag: A dict with 'parents' (list of lists, topologically ordered) and 'outputs' (list of output vertex indices)
        fast_memory_budget: Number of fast memory slots available
        
    Returns:
        List of [operation, vertex] where operation is one of 'load', 'store', 'compute', 'delete'
    """
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s margin for 90s wall-clock limit
    
    # Extract DAG structure
    parents = dag['parents']
    outputs = dag['outputs']
    n = len(parents)
    
    # Build reverse adjacency (children)
    children = [[] for _ in range(n)]
    for i in range(n):
        for p in parents[i]:
            if p is not None and p >= 0:
                children[p].append(i)
    
    # Identify input nodes (no parents) and output nodes
    input_nodes = [i for i in range(n) if len(parents[i]) == 0 or all(p is None or p < 0 for p in parents[i])]
    output_nodes = set(outputs)
    
    # Topological order is already given by the index order
    
    # Strategy: Use a layered pebbling approach
    # For layered DAGs, we process layer by layer
    # Key insight: minimize loads/stores by keeping intermediate results
    
    # Build a simple but effective strategy:
    # 1. Load all inputs
    # 2. Process each node in topological order
    # 3. For each node, ensure its parents are available (load if needed)
    # 4. Compute the node
    # 5. Decide whether to store the result or keep it in fast memory
    
    # Use a greedy strategy with fast memory management
    # Track which nodes are in fast memory (red pebbles)
    
    fast_mem = set()  # Nodes currently in fast memory (red)
    operations = []
    
    # Helper: get parent indices for a node
    def get_parents(node):
        ps = parents[node]
        return [p for p in ps if p is not None and p >= 0]
    
    # Helper: is node an input?
    def is_input(node):
        ps = get_parents(node)
        return len(ps) == 0
    
    # Helper: is node an output?
    def is_output(node):
        return node in output_nodes
    
    # Helper: get children of a node
    def get_children(node):
        return children[node]
    
    # Helper: count how many times a node's value is needed (fan-out)
    # Precompute fan-out for each node
    fanout = [len(children[i]) for i in range(n)]
    
    # Strategy: 
    # - Load inputs as needed
    # - Compute nodes in topological order
    # - Keep results in fast memory if budget allows and they'll be needed again
    # - Store results that won't be needed again or if budget is full
    
    # First pass: determine which nodes to keep in fast memory
    # Use a heuristic: keep nodes with high fan-out or that are outputs
    
    # Simple heuristic: keep a node in fast memory if:
    # 1. It's an output
    # 2. It has multiple children (high fan-out)
    # 3. Fast memory has space
    
    # Process nodes in topological order
    for node in range(n):
        # Check deadline
        if time.time() > deadline:
            break
        
        # Ensure all parents are available in fast memory
        ps = get_parents(node)
        for p in ps:
            if p not in fast_mem:
                # Need to load parent
                operations.append(['load', p])
                fast_mem.add(p)
        
        # Compute the node
        operations.append(['compute', node])
        fast_mem.add(node)
        
        # Decide whether to keep this node in fast memory or store it
        # Store if:
        # - Fast memory is full and this node has low priority
        # - This node won't be needed again (no remaining children that need it)
        
        # Check if this node is needed by any remaining unprocessed nodes
        needed_by_remaining = False
        for c in children[node]:
            if c > node:  # c is processed after node
                needed_by_remaining = True
                break
        
        # If fast memory is over budget, we need to evict something
        if len(fast_mem) > fast_memory_budget:
            # Find a node to evict (store)
            # Priority: evict nodes that are no longer needed, or low fan-out
            # Simple: evict the oldest non-output node that's no longer needed
            
            # Find candidates to evict
            candidates = []
            for fm_node in list(fast_mem):
                if fm_node == node:
                    continue  # Don't evict the just-computed node immediately
                # Check if fm_node is still needed
                fm_needed = False
                for c in children[fm_node]:
                    if c > node:  # c is processed after current node
                        fm_needed = True
                        break
                if not fm_needed and not is_output(fm_node):
                    candidates.append(fm_node)
            
            # If we have candidates, evict one
            if candidates:
                # Evict the one with lowest fan-out among candidates
                evict_node = min(candidates, key=lambda x: fanout[x])
                operations.append(['store', evict_node])
                fast_mem.discard(evict_node)
            else:
                # No good candidates, evict the oldest non-current node
                # Just pick the first one that's not an output
                for fm_node in list(fast_mem):
                    if fm_node != node and not is_output(fm_node):
                        operations.append(['store', fm_node])
                        fast_mem.discard(fm_node)
                        break
        
        # If this node is not needed by remaining nodes and not an output, consider storing it
        if not needed_by_remaining and not is_output(node):
            # Store it to free up space
            operations.append(['store', node])
            fast_mem.discard(node)
    
    # Ensure all outputs are in fast memory at the end
    for out in output_nodes:
        if out not in fast_mem:
            operations.append(['load', out])
            fast_mem.add(out)
    
    # Verify operation count
    if len(operations) > 20000:
        # Truncate if necessary (shouldn't happen for reasonable DAGs)
        operations = operations[:20000]
    
    return operations

# EVOLVE-BLOCK-END
