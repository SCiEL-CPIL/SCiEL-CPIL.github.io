"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    N = 256
    # Precompute conflict matrix
    # conflict[i][j] is True if i and j cannot be in the same set
    # i, j are 1..256. We use 0-indexed for internal list, so index i-1.
    conflict = [[False] * N for _ in range(N)]
    # Also precompute list of neighbors for each node for faster degree calculation
    neighbors = [[] for _ in range(N)]
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                conflict[i-1][j-1] = True
                conflict[j-1][i-1] = True
                neighbors[i-1].append(j-1)
                neighbors[j-1].append(i-1)

    def greedy_mis_min_degree():
        """
        Greedy Max Independent Set using Minimum Degree heuristic.
        1. Start with all nodes available.
        2. Pick node with min degree in the remaining subgraph.
        3. Add to set.
        4. Remove node and its neighbors from available.
        """
        available = set(range(N))
        selected = []
        
        while available:
            # Find node with minimum degree within 'available'
            min_deg = float('inf')
            best_node = -1
            # Optimization: if we have many nodes, this is O(N^2) per step, total O(N^3).
            # N=256, N^3 ~ 16M operations. Python is slow, but 16M simple ops is ~1-2s.
            # We can optimize by maintaining degrees dynamically, but let's try simple first.
            for node in available:
                deg = 0
                for nei in neighbors[node]:
                    if nei in available:
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
                    best_node = node
            
            if best_node == -1:
                break
                
            selected.append(best_node + 1)
            
            # Remove best_node and its neighbors from available
            to_remove = {best_node}
            for nei in neighbors[best_node]:
                if nei in available:
                    to_remove.add(nei)
            
            available -= to_remove
            
        return selected

    def local_improve(selected):
        """
        Try to add any node not in selected that has no conflicts with selected.
        """
        sel_set = set(selected)
        improved = True
        while improved:
            improved = False
            for node in range(N):
                if node not in sel_set:
                    # Check if node conflicts with any in sel_set
                    compatible = True
                    for sel in sel_set:
                        if conflict[node][sel]:
                            compatible = False
                            break
                    if compatible:
                        sel_set.add(node)
                        improved = True
                        break
        return list(sel_set)

    best_set = []
    
    # Run Greedy MIS with different tie-breaking strategies
    # Strategy 1: Min Degree, then Smallest Node
    # Strategy 2: Min Degree, then Largest Node
    # We can modify the greedy function to accept a tie-breaker, 
    # but for simplicity, we run the base version and a variant.
    
    # To allow different tie-breaking, we can shuffle the 'available' set logic or 
    # just rely on the fact that `min` picks the first one found.
    # Let's create a version that iterates through nodes in a specific order when degrees are equal.
    
    def greedy_mis_tiebreak(order_func):
        available = set(range(N))
        selected = []
        
        while available:
            min_deg = float('inf')
            candidates = []
            for node in available:
                deg = 0
                for nei in neighbors[node]:
                    if nei in available:
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
                    candidates = [node]
                elif deg == min_deg:
                    candidates.append(node)
            
            if not candidates:
                break
            
            # Apply tie-breaker
            best_node = order_func(candidates)
            
            selected.append(best_node + 1)
            
            to_remove = {best_node}
            for nei in neighbors[best_node]:
                if nei in available:
                    to_remove.add(nei)
            
            available -= to_remove
            
        return selected

    # Tie-breaker 1: Smallest node
    def tb_min(cands):
        return min(cands)

    # Tie-breaker 2: Largest node
    def tb_max(cands):
        return max(cands)

    # Tie-breaker 3: Random (but deterministic seed for reproducibility in this run)
    # We'll just use min and max, and maybe a few random shuffles of the initial node list 
    # if we implemented it differently, but here we use the set logic.
    # To add randomness to tie-breaking, we can sort candidates by a random key.
    
    best_size = 0
    best_candidate = []
    
    # Run deterministic strategies
    for _ in range(20):
        s1 = greedy_mis_tiebreak(tb_min)
        s1 = local_improve(s1)
        if len(s1) > best_size:
            best_size = len(s1)
            best_candidate = s1
            
        s2 = greedy_mis_tiebreak(tb_max)
        s2 = local_improve(s2)
        if len(s2) > best_size:
            best_size = len(s2)
            best_candidate = s2

    # Add some randomized tie-breaking restarts
    for _ in range(50):
        def tb_random(cands):
            return random.choice(cands)
        
        s3 = greedy_mis_tiebreak(tb_random)
        s3 = local_improve(s3)
        if len(s3) > best_size:
            best_size = len(s3)
            best_candidate = s3

    return best_candidate
# EVOLVE-BLOCK-END
