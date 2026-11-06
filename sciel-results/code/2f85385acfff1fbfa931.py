# EVOLVE-BLOCK-START
import time
import sys
sys.setrecursionlimit(10000)

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85  # seconds, leave 5s margin for overhead/return
    
    # Extract structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        parents = dag.get('parents', {})
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
    else:
        vertices = getattr(dag, 'vertices', [])
        parents = getattr(dag, 'parents', {})
        inputs = set(getattr(dag, 'inputs', []))
        outputs = set(getattr(dag, 'outputs', []))
    
    if not vertices:
        return []
        
    # Build children map and reverse topological order (post-order) for pebbling
    # Actually, we need topological order for computation.
    children = {v: [] for v in vertices}
    for v in vertices:
        if v in parents:
            for p in parents[v]:
                if p in children:
                    children[p].append(v)
    
    # Identify essential vertices: those that are ancestors of any output
    essential = set(outputs)
    changed = True
    while changed:
        changed = False
        for v in vertices:
            if v not in essential:
                for c in children[v]:
                    if c in essential:
                        essential.add(v)
                        changed = True
                        break
    
    # Greedy Pebbling
    in_fast = set()
    operations = []
    
    # We need to compute vertices in topological order.
    # Let's find topological order using Kahn's algorithm or DFS.
    # Since parents are given, we can assume vertices are listed in some order, 
    # but to be safe, let's compute in-degree.
    in_degree = {v: 0 for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in in_degree:
                in_degree[v] += 1
                
    from collections import deque
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    # If topo_order doesn't contain all vertices, there's a cycle or disconnected part.
    # Assume valid DAG.
    
    def evict_one(exclude=None):
        """Evict one vertex from fast memory."""
        if len(in_fast) == 0:
            return
        candidates = [v for v in in_fast if v not in (exclude or set())]
        if not candidates:
            return
        # Heuristic: Evict non-essential first. Then essential with fewest essential children.
        # Calculate "importance": count of essential descendants
        # For speed, we just use: 1 if not essential, 0 if essential.
        # Then among essentials, prefer those with fewer children in essential set?
        # Let's keep it simple: 
        # 1. Not essential
        # 2. Essential but no essential children (leaves of essential subgraph)
        # 3. Essential with essential children
        
        def importance(v):
            if v not in essential:
                return 0
            # Count essential children
            ess_children = sum(1 for c in children[v] if c in essential)
            # If it has no essential children, it's a leaf in essential subgraph, less important to keep?
            # Actually, we need it to be available if its parent is recomputed? No, if we evict it, we must recompute it.
            # Recomputing requires its parents.
            # So vertices closer to inputs are more expensive to recompute?
            # Let's just evict non-essential first.
            return 1
            
        best_v = min(candidates, key=lambda v: (importance(v), len(children[v])))
        in_fast.remove(best_v)
        operations.append(['delete', best_v])
        
    def ensure_in_fast(v):
        """Ensure vertex v is in fast memory."""
        if v in in_fast:
            return True
        
        # If budget is full, evict
        if len(in_fast) >= fast_memory_budget:
            evict_one(exclude={v})
            # If still full (budget 0?), we can't do anything. Assume budget > 0.
            if len(in_fast) >= fast_memory_budget:
                # Try to evict anything, even if v is excluded? No, we need space for v.
                # If budget is 0 and v is not input, impossible.
                # If budget is 1 and in_fast has 1, evict that one.
                # The exclude={v} prevents evicting v if it was already there, but it's not.
                # So we evict some other vertex.
                evict_one(exclude=set()) # Remove exclusion to force eviction if needed
                if len(in_fast) >= fast_memory_budget:
                    return False # Failed to make space
        
        if v in inputs:
            operations.append(['load', v])
            in_fast.add(v)
            return True
            
        # Compute v
        # Ensure parents are in fast memory
        for p in parents.get(v, []):
            if not ensure_in_fast(p):
                return False
        
        # Now parents are in fast memory.
        # Ensure space for v
        if len(in_fast) >= fast_memory_budget:
            evict_one(exclude={v})
            if len(in_fast) >= fast_memory_budget:
                return False
        
        operations.append(['compute', v])
        in_fast.add(v)
        return True
        
    # Process vertices in topological order
    # We only care about essential vertices for the final state, 
    # but we might need to compute non-essentials if they are parents of essentials.
    # So we process all vertices in topo order.
    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            break
        # We don't necessarily need to keep non-essential vertices in fast memory at the end,
        # but we need them to compute their children.
        # ensure_in_fast will load/compute it.
        ensure_in_fast(v)
        
    # Final step: Ensure all outputs are in fast memory
    # The loop above might have evicted some outputs or their ancestors.
    # We need to bring them back.
    for out in outputs:
        if time.time() - start_time > DEADLINE:
            break
        if out not in in_fast:
            ensure_in_fast(out)
            
    # Verify all outputs are in fast memory.
    # If not, we have a problem. But ensure_in_fast should have handled it.
    # If budget is too small to hold all outputs, this might fail.
    # Assuming feasible instances.
    
    return operations
# EVOLVE-BLOCK-END
