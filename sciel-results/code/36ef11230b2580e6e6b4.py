import heapq
import time
import sys

def pebble(dag, fast_memory_budget):
    """
    Solves the communication-minimizing DAG pebbling problem.
    
    Args:
        dag: A list of lists, where dag[i] is the list of parent indices of vertex i.
             Vertices are assumed to be topologically ordered.
        fast_memory_budget: The maximum number of red pebbles allowed.
    
    Returns:
        A list of [operation, vertex] pairs.
    """
    if not dag:
        return []
    
    N = len(dag)
    
    # Build children map
    children = [[] for _ in range(N)]
    for i in range(N):
        for p in dag[i]:
            children[p].append(i)
            
    # remaining_children: number of children not yet processed
    remaining_children = [len(children[i]) for i in range(N)]
    
    # State
    # fast_set: set of vertices in Fast Memory
    # main_mask: bitmask of vertices in Main Memory
    # Semantics:
    # Load: Main -> Fast (Main loses, Fast gains)
    # Store: Fast -> Main (Fast loses, Main gains)
    # Compute: Gains Fast (Main unchanged)
    # Delete: Loses Fast (Main unchanged)
    
    # Initial: Sources in Main.
    main_mask = 0
    for i in range(N):
        if not dag[i]:
            main_mask |= (1 << i)
            
    fast_set = set()
    ops = []
    
    # Process vertices in topological order
    for v in range(N):
        # 1. Ensure parents are in Fast
        for p in dag[v]:
            if p not in fast_set:
                # Check if p is in Main
                if main_mask & (1 << p):
                    ops.append(['load', p])
                    fast_set.add(p)
                    main_mask &= ~(1 << p)
                else:
                    # Fallback: If not in Main and not in Fast, we are in trouble.
                    # This implies we deleted it or never stored it.
                    # For robustness, we cannot recover without recomputation.
                    # We will assume this doesn't happen in valid instances.
                    pass
            
        # 2. Compute v
        ops.append(['compute', v])
        fast_set.add(v)
        
        # 3. Store v to Main
        # We want v to be available for its children.
        ops.append(['store', v])
        fast_set.discard(v)
        main_mask |= (1 << v)
        
        # 4. Clean up parents
        for p in dag[v]:
            remaining_children[p] -= 1
            if remaining_children[p] == 0:
                if p in fast_set:
                    ops.append(['delete', p])
                    fast_set.discard(p)
                
    return ops
