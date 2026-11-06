import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n = 20
    # Initialize with complete graph
    # adj[i] is a bitmask of neighbors
    adj = [(1 << n) - 1 for _ in range(n)]
    for i in range(n):
        adj[i] &= ~(1 << i)
    
    edges_count = n * (n - 1) // 2
    
    def get_violation_score():
        score = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                count = common.bit_count()
                if count > 2:
                    score += count - 2
        return score
    
    current_score = get_violation_score()
    
    # Greedy removal with randomization
    while current_score > 0:
        if time.time() > deadline:
            break
            
        # Find all violating pairs
        violating_pairs = []
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                count = common.bit_count()
                if count > 2:
                    violating_pairs.append((u, v, common))
        
        if not violating_pairs:
            break
            
        # Pick a random violating pair to focus on, or the one with max common neighbors
        # To be more robust, pick random from top violators
        violating_pairs.sort(key=lambda x: x[2].bit_count(), reverse=True)
        # Pick a random one from the top 5 or just random
        u, v, common = violating_pairs[random.randint(0, min(4, len(violating_pairs) - 1))]
        
        # Candidates: remove (w, u) or (w, v) for w in common
        candidates = []
        common_list = [i for i in range(n) if (common >> i) & 1]
        for w in common_list:
            candidates.append((u, w))
            candidates.append((v, w))
        
        # Evaluate each candidate
        best_move = None
        best_score = current_score + 1000 # Penalize non-improving moves heavily? No, we might need to move sideways.
        
        # We want to minimize the new score.
        # If no move reduces score, we pick a random move among candidates to escape.
        
        moves_evaluated = []
        for a, b in candidates:
            # Remove edge (a, b)
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            
            new_score = get_violation_score()
            moves_evaluated.append((new_score, a, b))
            
            # Restore edge (a, b)
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
            
        # Find the move with minimum score
        moves_evaluated.sort(key=lambda x: x[0])
        
        # If the best move improves the score, take it.
        # Otherwise, take a random move from the top few (or just random candidate) to explore.
        if moves_evaluated[0][0] < current_score:
            new_score, a, b = moves_evaluated[0]
        else:
            # Pick random from top 3 or just random candidate
            # To avoid cycling, maybe random from all candidates
            new_score, a, b = moves_evaluated[random.randint(0, len(moves_evaluated) - 1)]
        
        # Apply the move
        adj[a] &= ~(1 << b)
        adj[b] &= ~(1 << a)
        edges_count -= 1
        current_score = new_score
        
    # Extract edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
