import time
import sys
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    
    # Parse DAG structure
    # Assume dag is a list of lists of parents.
    # If dag is a dict, convert to list.
    if isinstance(dag, dict):
        # Try to infer vertices
        keys = list(dag.keys())
        # This is ambiguous. Let's assume list of lists is more likely for "topologically ordered parents".
        # If dict, maybe keys are vertices and values are parents.
        # We'll handle list of lists primarily.
        pass
    
    if not isinstance(dag, list):
        # Fallback: try to handle other types, but assume list of lists
        # If this fails, return empty or raise
        return []

    n = len(dag)
    if n == 0:
        return []

    # Identify inputs and outputs
    is_input = [len(parents) == 0 for parents in dag]
    
    # Build children list
    children = defaultdict(list)
    for i, parents in enumerate(dag):
        for p in parents:
            children[p].append(i)
            
    is_output = [len(children[i]) == 0 for i in range(n)]
    
    # Topological Sort
    in_degree = [len(p) for p in dag]
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    top_order = []
    while queue:
        u = queue.popleft()
        top_order.append(u)
        for v in children[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(top_order) != n:
        # Cycle? Should not happen for DAG.
        # Return naive fallback
        return []

    # If DAG fits in memory, use optimal strategy
    if n <= fast_memory_budget:
        ops = []
        # Load all inputs
        for i in range(n):
            if is_input[i]:
                ops.append(['load', i])
        
        # Compute all in topological order
        for i in top_order:
            if not is_input[i]: # Inputs are already "computed" by loading? 
                # Wait, "Inputs start blue". Load puts them in red.
                # Compute operation: "compute".
                # Does loading an input count as computing it?
                # "Inputs start blue... operations load/store/compute/delete".
                # Usually, load makes it available. Compute calculates it from parents.
                # Inputs have no parents. So they don't need compute.
                # But do they need to be "computed" to be valid?
                # "Every red value consumes one fast-memory slot, including newly computed results."
                # If I load an input, it's red. Is it "computed"?
                # The prompt says "compute" is an operation.
                # If an input is loaded, it is in memory.
                # If I try to compute it, it has no parents.
                # Let's assume inputs do NOT need a 'compute' op, just 'load'.
                # Internal nodes need 'compute'.
                pass
        
        # Correct Logic for Small DAG:
        # 1. Load Inputs.
        # 2. Compute Internal Nodes (in topo order).
        # 3. Store Outputs.
        
        ops = []
        in_mem = set()
        
        for i in top_order:
            if is_input[i]:
                if i not in in_mem:
                    ops.append(['load', i])
                    in_mem.add(i)
            else:
                # Parents must be in mem. Since n <= budget, they are.
                ops.append(['compute', i])
                in_mem.add(i)
        
        for i in range(n):
            if is_output[i]:
                if i in in_mem:
                    ops.append(['store', i])
                    in_mem.remove(i)
                    
        return ops

    # Heuristic for Large DAG
    ops = []
    in_mem = set()
    processed = set()
    
    # remaining_uses: how many children of v are NOT yet processed
    remaining_uses = [0] * n
    for i in range(n):
        remaining_uses[i] = len(children[i])
        
    # We process in topological order
    for v in top_order:
        # Deadline check
        if time.time() - start_time > 50:
            # Emergency exit: store all in mem to be safe? 
            # Or just return what we have? 
            # The evaluator verifies exact state. 
            # If we stop early, we might not have computed all outputs.
            # This will fail verification.
            # Better to continue? 
            # If we hit deadline, we return current ops. 
            # If it's invalid, score is 0. 
            # So we must ensure we finish.
            pass

        # Check if v is already in memory
        if v in in_mem:
            # It's ready
            pass
        else:
            # Need to bring parents into memory
            for p in dag[v]:
                if p not in in_mem:
                    # Check memory capacity
                    if len(in_mem) >= fast_memory_budget:
                        # Evict a vertex
                        # Find victim: min remaining_uses
                        # If tie, prefer higher topological index (closer to end)
                        # Or just min remaining_uses
                        victim = None
                        min_uses = float('inf')
                        # We can only evict vertices that are NOT needed for the current computation?
                        # Actually, we can evict anything, but we must be able to reload/recompute.
                        # If we evict a parent of v, we just loaded it, so it's in mem.
                        # We are evicting BEFORE loading p.
                        # So p is not in mem.
                        # We need space for p.
                        
                        # Candidate victims: all u in in_mem
                        for u in in_mem:
                            # u must be evictable
                            # If u is an input, we can store it.
                            # If u is computed, we can store it.
                            # We cannot delete u if it's an input (can't recompute).
                            # We can delete u if it's computed, but then we must recompute later.
                            # For simplicity, we always STORE.
                            
                            # Heuristic: Evict u with min remaining_uses
                            if remaining_uses[u] < min_uses:
                                min_uses = remaining_uses[u]
                                victim = u
                                
                        if victim is not None:
                            ops.append(['store', victim])
                            in_mem.remove(victim)
                        else:
                            # Should not happen if budget >= 1
                            pass

                    # Load p
                    ops.append(['load', p])
                    in_mem.add(p)
                    
        # Now all parents of v are in in_mem
        # Compute v
        if not is_input[v]:
            ops.append(['compute', v])
        in_mem.add(v)
        processed.add(v)
        
        # Update remaining_uses for parents of v
        for p in dag[v]:
            remaining_uses[p] -= 1
            
    # Finalize: Store all outputs
    for i in range(n):
        if is_output[i]:
            if i in in_mem:
                ops.append(['store', i])
                in_mem.remove(i)
            else:
                # If output is not in mem, it must have been stored earlier?
                # If it was stored, it's blue. That's fine.
                # But wait, if it was stored, it's blue.
                # If it was never stored, it's in mem (red).
                # We need it to be blue.
                # So if it's in mem, store it.
                # If it's not in mem, it's already blue (stored or deleted).
                # If it was deleted, it's lost. But we need the value.
                # Our heuristic always stores, so it should be in slow memory.
                pass
                
    return ops

# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    # Wrapper to handle potential import issues or context
    # The task says "Keep the function names and signatures of the seed program".
    # The seed program called pebble_baseline. 
    # We are replacing the logic entirely.
    try:
        return _pebble_impl(dag, fast_memory_budget)
    except Exception:
        # Fallback to naive if possible? 
        # Or return empty?
        # Better to try a very simple naive approach if impl fails
        return []

def _pebble_impl(dag, fast_memory_budget):
    # Move the logic here to keep pebble clean
    import time
    from collections import defaultdict, deque
    
    start_time = time.time()
    
    if not isinstance(dag, list):
        return []
        
    n = len(dag)
    if n == 0:
        return []

    is_input = [len(parents) == 0 for parents in dag]
    
    children = defaultdict(list)
    for i, parents in enumerate(dag):
        for p in parents:
            children[p].append(i)
            
    is_output = [len(children[i]) == 0 for i in range(n)]
    
    in_degree = [len(p) for p in dag]
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    top_order = []
    while queue:
        u = queue.popleft()
        top_order.append(u)
        for v in children[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(top_order) != n:
        return []

    if n <= fast_memory_budget:
        ops = []
        in_mem = set()
        
        for i in top_order:
            if is_input[i]:
                if i not in in_mem:
                    ops.append(['load', i])
                    in_mem.add(i)
            else:
                ops.append(['compute', i])
                in_mem.add(i)
        
        for i in range(n):
            if is_output[i]:
                if i in in_mem:
                    ops.append(['store', i])
                    in_mem.remove(i)
                    
        return ops

    ops = []
    in_mem = set()
    processed = set()
    
    remaining_uses = [0] * n
    for i in range(n):
        remaining_uses[i] = len(children[i])
        
    for v in top_order:
        if time.time() - start_time > 55:
            # If we are close to timeout, we might not finish.
            # But we must return something.
            pass

        if v not in in_mem:
            for p in dag[v]:
                if p not in in_mem:
                    if len(in_mem) >= fast_memory_budget:
                        victim = None
                        min_uses = float('inf')
                        for u in in_mem:
                            if remaining_uses[u] < min_uses:
                                min_uses = remaining_uses[u]
                                victim = u
                                
                        if victim is not None:
                            ops.append(['store', victim])
                            in_mem.remove(victim)
                    
                    ops.append(['load', p])
                    in_mem.add(p)
                    
        if not is_input[v]:
            ops.append(['compute', v])
        in_mem.add(v)
        processed.add(v)
        
        for p in dag[v]:
            remaining_uses[p] -= 1
            
    for i in range(n):
        if is_output[i]:
            if i in in_mem:
                ops.append(['store', i])
                in_mem.remove(i)
                
    return ops

# EVOLVE-BLOCK-END
