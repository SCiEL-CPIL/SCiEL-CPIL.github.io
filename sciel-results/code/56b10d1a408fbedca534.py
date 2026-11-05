<<<<<<< SEARCH
import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85.0
    
    parents = dag['parents']
    outputs = set(dag.get('outputs', []))
    num_vertices = len(parents)
    
    children = [[] for _ in range(num_vertices)]
    for i in range(num_vertices):
        for p in parents[i]:
            children[p].append(i)
    
    inputs = [i for i in range(num_vertices) if len(parents[i]) == 0]
    
    # Track which vertices are in fast memory
    in_fast = set(inputs)
    # Pending children count for each vertex
    pending = {v: len(children[v]) for v in range(num_vertices)}
    
    ops = []
    
    for v in range(num_vertices):
        if time.time() > deadline:
            break
        
        # Load parents if not in fast memory
        for p in parents[v]:
            if p not in in_fast:
                ops.append(['load', p])
                in_fast.add(p)
        
        # Compute v
        ops.append(['compute', v])
        in_fast.add(v)
        
        # Decrement pending for parents, delete if no more children and not output/input
        for p in parents[v]:
            pending[p] -= 1
            if pending[p] == 0 and p not in outputs and p not in inputs:
                if p in in_fast:
                    ops.append(['delete', p])
                    in_fast.discard(p)
    
    # Ensure outputs are in fast memory
    for v in sorted(outputs):
        if v not in in_fast:
            ops.append(['load', v])
            in_fast.add(v)
    
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
=======
import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    parents = dag['parents']
    outputs = set(dag.get('outputs', []))
    num_vertices = len(parents)
    
    children = [[] for _ in range(num_vertices)]
    for i in range(num_vertices):
        for p in parents[i]:
            children[p].append(i)
    
    # in_fast tracks vertices currently in fast memory
    in_fast = set()
    
    ops = []
    
    # Helper to check if we can add to fast memory
    # We need to evict if full.
    def try_add(v):
        nonlocal in_fast
        if v in in_fast:
            return
        if len(in_fast) < fast_memory_budget:
            in_fast.add(v)
            ops.append(['load', v])
        else:
            # Need to evict someone
            # Strategy: Evict a vertex that is not an output and has no remaining children
            # 'remaining children' means children with index > current processing index
            # Since we process in topological order (0 to N-1),
            # a vertex u is "done" if all its children have index < current_v
            # We are currently trying to load for current_v.
            # So we can evict u if max(children[u]) < current_v and u not in outputs
            evicted = False
            # Prioritize evicting non-outputs
            for u in list(in_fast):
                if u in outputs:
                    continue
                # Check if u has any children with index >= current_v
                has_future_child = False
                for c in children[u]:
                    if c >= current_v:
                        has_future_child = True
                        break
                if not has_future_child:
                    ops.append(['delete', u])
                    in_fast.discard(u)
                    evicted = True
                    break
            
            if evicted:
                in_fast.add(v)
                ops.append(['load', v])
            else:
                # Critical failure: cannot fit parent
                # This should not happen if budget >= 1 and we manage well, 
                # but if budget is 0, we can't do anything.
                # If budget is too small, the task might be unsolvable or require recomputation.
                # For now, if we can't load, we just skip? No, compute will fail.
                # We must ensure budget >= 1 for any non-trivial DAG.
                # If we can't evict, we are stuck.
                pass

    # Process vertices in topological order
    for current_v in range(num_vertices):
        if time.time() > deadline:
            break
            
        # 1. Load parents
        for p in parents[current_v]:
            if p not in in_fast:
                try_add(p)
        
        # 2. Compute current_v
        # Note: If a parent was not loaded (due to budget issues), this is invalid.
        # But try_add should have handled it.
        ops.append(['compute', current_v])
        
        # Add current_v to fast memory?
        # If current_v is an output, it MUST be in fast memory at the end.
        # If current_v is not an output, it can be deleted later.
        # However, we need its value to be "blue" if it's an output.
        # The problem says "every red value consumes one fast-memory slot".
        # "Inputs start blue". "Outputs must finish blue".
        # Usually, 'compute' produces a blue value in fast memory.
        # So after compute, current_v is in fast memory.
        
        if current_v not in in_fast:
             # If we didn't load it (e.g. if it was an input and we didn't load it initially? 
             # No, inputs are loaded when needed as parents or if they are outputs?
             # Wait, inputs are vertices with no parents.
             # If an input is an output, it must be blue.
             # If an input is not an output, it doesn't need to be blue at the end, 
             # but it needs to be blue when used as a parent.
             
             # If current_v is an input, it was "blue" initially? 
             # "Inputs start blue". This usually means they are available in slow memory, 
             # and loading them makes them blue in fast memory.
             # If current_v is an input, we didn't load it in step 1 because it has no parents.
             # But if it's an output, we need it blue at the end.
             # If it's not an output, we don't care.
             
             # If current_v is an input and we are computing it? 
             # "Compute" usually implies combining parents. Inputs have no parents.
             # So we don't "compute" inputs. We just "load" them if they are used.
             # But the loop `for p in parents[current_v]` does nothing for inputs.
             # So `ops.append(['compute', current_v])` is WRONG for inputs.
             
             # Correction: Only compute non-inputs.
             # Inputs are just loaded.
             
             # Let's re-structure.
             pass
             
        # Actually, let's restart the logic inside the loop properly.
        
    # The above inline logic got messy. Let's replace the whole function body with a clean implementation.
    pass
>>>>>>> REPLACE

The previous edit attempt was too messy and incomplete. I will provide a clean, complete replacement for the `pebble` function that correctly handles:
1.  Topological order.
2.  Distinction between Inputs (no parents) and Intermediate/Output vertices.
3.  Strict Fast Memory Budget enforcement with eviction of "done" non-output vertices.
4.  Finalization of outputs.

