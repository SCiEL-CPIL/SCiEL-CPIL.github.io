import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    def is_valid_path(path):
        if not path or len(path) < 1:
            return False
        seen = set(path)
        if len(seen) != len(path):
            return False
        for i in range(len(path) - 1):
            if path[i + 1] not in adj[path[i]]:
                return False
        return True
        
    def greedy_path(start, strategy='min_degree'):
        path = [start]
        seen = {start}
        while True:
            candidates = adj[path[-1]] - seen
            if not candidates:
                break
            if strategy == 'min_degree':
                v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            elif strategy == 'max_degree':
                v = max(candidates, key=lambda w: (len(adj[w] - seen), -w))
            elif strategy == 'random':
                v = random.choice(list(candidates))
            else:
                v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            path.append(v)
            seen.add(v)
        return path
        
    def try_extend(path, adj, seen):
        changed = True
        while changed:
            changed = False
            if path:
                last = path[-1]
                candidates = adj[last] - seen
                if candidates:
                    v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
                    path.append(v)
                    seen.add(v)
                    changed = True
            if path:
                first = path[0]
                candidates = adj[first] - seen
                if candidates:
                    v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
                    path.insert(0, v)
                    seen.add(v)
                    changed = True
        return path
        
    def local_improve(path, adj, max_iters=1000):
        best = list(path)
        best_len = len(best)
        seen = set(best)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            
            # 1. Try to extend from ends
            extended = try_extend(best, adj, set(best))
            if len(extended) > best_len:
                best = extended
                best_len = len(best)
                seen = set(best)
                improved = True
                continue # Restart loop after extension
                
            # 2. 2-Opt Move: Reverse a segment to potentially unlock extensions
            # We accept 2-Opt even if length doesn't change, to explore configuration space
            # But we only do it if it doesn't break the path.
            # To avoid infinite loops of 2-Opt, we can limit or randomize.
            # Here we try a random 2-Opt.
            if len(best) > 2:
                i = random.randint(0, len(best) - 2)
                j = random.randint(i + 1, len(best) - 1)
                
                # Check validity of reversal
                valid = True
                if i > 0 and best[j] not in adj[best[i-1]]:
                    valid = False
                if valid and j < len(best) - 1 and best[i] not in adj[best[j+1]]:
                    valid = False
                    
                if valid:
                    new_path = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    # Check if this new configuration allows extension
                    new_seen = set(new_path)
                    # Simple heuristic: accept if it allows at least one new extension
                    # Or just accept it as a neutral move in SA-like fashion
                    # For local_improve, let's accept if it leads to immediate extension
                    # Or just accept it to shuffle.
                    # Let's accept it and see if it helps.
                    best = new_path
                    # Don't update best_len, it's the same
                    improved = True # Force another iteration to try extension
                
            if not improved:
                break
                
        return best
        
    def anneal_path(initial_path, adj, iters=50000):
        current = list(initial_path)
        current_seen = set(current)
        best = list(current)
        best_seen = set(current)
        T = 1.0
        cooling = 0.9995
        rng = random.Random(42)
        
        for _ in range(iters):
            if time.time() > deadline:
                break
                
            move_type = rng.random()
            
            # Move 1: Insert unvisited vertex (Delta = +1)
            if move_type < 0.4:
                unvisited = set(range(n)) - current_seen
                if unvisited:
                    v = rng.choice(list(unvisited))
                    valid_positions = []
                    for i in range(len(current) + 1):
                        if i == 0:
                            if i < len(current) and v in adj[current[0]]:
                                valid_positions.append(i)
                        elif i == len(current):
                            if i > 0 and v in adj[current[-1]]:
                                valid_positions.append(i)
                        else:
                            if v in adj[current[i-1]] and v in adj[current[i]]:
                                valid_positions.append(i)
                    
                    if valid_positions:
                        pos = rng.choice(valid_positions)
                        new_path = current[:pos] + [v] + current[pos:]
                        current = new_path
                        current_seen.add(v)
                        if len(current) > len(best):
                            best = list(current)
                            best_seen = set(current)
                            T = 1.0 # Reset temperature on improvement
                            
            # Move 2: 2-Opt (Reverse segment) (Delta = 0)
            elif move_type < 0.8:
                if len(current) > 2:
                    i = rng.randint(0, len(current) - 2)
                    j = rng.randint(i + 1, len(current) - 1)
                    
                    valid = True
                    if i > 0 and current[j] not in adj[current[i-1]]:
                        valid = False
                    if valid and j < len(current) - 1 and current[i] not in adj[current[j+1]]:
                        valid = False
                    
                    if valid:
                        new_path = current[:i] + current[i:j+1][::-1] + current[j+1:]
                        # Accept neutral move with high probability to explore
                        # Standard SA accepts neutral moves with probability 1
                        # Or we can be selective. Let's accept it.
                        current = new_path
                        # current_seen remains the same
                        
            # Move 3: Delete vertex (Delta = -1)
            else:
                if len(current) > 2:
                    i = rng.randint(1, len(current) - 2)
                    v = current[i]
                    if current[i-1] in adj[current[i+1]]:
                        new_path = current[:i] + current[i+1:]
                        # Accept with probability exp(delta/T) = exp(-1/T)
                        if rng.random() < pow(2.718, -1.0 / T):
                            current = new_path
                            current_seen.remove(v)
                            
            T *= cooling
            
        return best
        
    # Phase 1: Generate candidates
    candidates = []
    for start in range(n):
        for strategy in ['min_degree', 'max_degree']:
            path = greedy_path(start, strategy)
            if path:
                candidates.append(path)
                
    rng = random.Random(42)
    for _ in range(100):
        start = rng.randint(0, n - 1)
        path = greedy_path(start, 'random')
        if path:
            candidates.append(path)
            
    candidates.sort(key=lambda p: len(p), reverse=True)
    
    best_path = candidates[0] if candidates else [0]
    best_len = len(best_path)
    
    # Phase 2: Local Improve top candidates
    for path in candidates[:20]:
        if len(path) <= best_len - 5:
            continue
        improved = local_improve(path, adj, max_iters=500)
        if len(improved) > best_len:
            best_path = improved
            best_len = len(improved)
        if time.time() > deadline:
            break
            
    # Phase 3: Simulated Annealing with 2-Opt
    if best_path and len(best_path) < n:
        annealed = anneal_path(best_path, adj, iters=20000)
        if len(annealed) > best_len:
            best_path = annealed
            best_len = len(best_path)
            
    # Final validation
    if not is_valid_path(best_path):
        best_path = greedy_path(0, 'min_degree')
        
    return best_path
# EVOLVE-BLOCK-END
