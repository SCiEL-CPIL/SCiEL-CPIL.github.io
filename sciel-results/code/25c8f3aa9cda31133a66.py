import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_set = []
    best_size = 0
    
    def get_initial_greedy():
        """Generate initial B2[2] set using different greedy strategies."""
        strategy = random.choice(['random', 'sorted', 'structured'])
        
        if strategy == 'random':
            candidates = list(range(1, 1001))
            random.shuffle(candidates)
        elif strategy == 'sorted':
            candidates = list(range(1, 1001))
            random.shuffle(candidates)
            candidates.sort(key=lambda x: random.random())
        else:  # structured - mix of small and large numbers
            small = list(range(1, 501))
            large = list(range(501, 1001))
            random.shuffle(small)
            random.shuffle(large)
            candidates = []
            i, j = 0, 0
            while i < len(small) or j < len(large):
                if i < len(small) and (j >= len(large) or random.random() < 0.5):
                    candidates.append(small[i])
                    i += 1
                else:
                    candidates.append(large[j])
                    j += 1
        
        S = []
        S_set = set()
        counts = {}
        
        for x in candidates:
            if x in S_set:
                continue
            
            # Check if x can be added
            valid = True
            # Check sums with existing elements
            for y in S:
                s = x + y
                if counts.get(s, 0) >= 2:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check double
            if counts.get(2 * x, 0) >= 2:
                continue
            
            # Add x
            S.append(x)
            S_set.add(x)
            
            # Update counts
            for y in S:
                if y == x:
                    break
                s = x + y
                counts[s] = counts.get(s, 0) + 1
            
            s_2x = 2 * x
            counts[s_2x] = counts.get(s_2x, 0) + 1
        
        return S, S_set, counts

    def local_search(S, S_set, counts):
        """Local search to improve B2[2] set."""
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Try to add any element
            # Use a precomputed list of candidates, shuffle once per iteration
            candidates = [x for x in range(1, 1001) if x not in S_set]
            if not candidates:
                break
            random.shuffle(candidates)
            
            added_any = False
            for x in candidates:
                # Check if x can be added
                valid = True
                for y in S:
                    s = x + y
                    if counts.get(s, 0) >= 2:
                        valid = False
                        break
                if not valid:
                    continue
                if counts.get(2 * x, 0) >= 2:
                    continue
                
                # Add x
                S.append(x)
                S_set.add(x)
                for y in S:
                    if y == x:
                        break
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                s_2x = 2 * x
                counts[s_2x] = counts.get(s_2x, 0) + 1
                
                improved = True
                added_any = True
                break  # Restart to check for more additions
            
            if added_any:
                continue
            
            # Phase 2: Try swaps if no additions possible
            if not S:
                break
            
            # Try a few random swaps
            for _ in range(20):
                if time.time() >= deadline:
                    break
                
                x = random.choice(S)
                y_candidates = [y for y in range(1, 1001) if y not in S_set]
                if not y_candidates:
                    break
                y = random.choice(y_candidates)
                
                # Temporarily remove x
                S.remove(x)
                S_set.remove(x)
                for z in S:
                    s = x + z
                    counts[s] -= 1
                    if counts[s] == 0:
                        del counts[s]
                s_2x = 2 * x
                counts[s_2x] -= 1
                if counts[s_2x] == 0:
                    del counts[s_2x]
                
                # Check if y can be added
                valid = True
                for z in S:
                    s = y + z
                    if counts.get(s, 0) >= 2:
                        valid = False
                        break
                if not valid:
                    if counts.get(2 * y, 0) >= 2:
                        valid = False
                
                if valid:
                    # Add y
                    S.append(y)
                    S_set.add(y)
                    for z in S:
                        if z == y:
                            break
                        s = y + z
                        counts[s] = counts.get(s, 0) + 1
                    s_2y = 2 * y
                    counts[s_2y] = counts.get(s_2y, 0) + 1
                    improved = True
                    break
                else:
                    # Revert: add x back
                    S.append(x)
                    S_set.add(x)
                    for z in S:
                        if z == x:
                            break
                        s = x + z
                        counts[s] = counts.get(s, 0) + 1
                    s_2x = 2 * x
                    counts[s_2x] = counts.get(s_2x, 0) + 1
        
        return S, S_set, counts

    # Main loop with multiple restarts
    num_restarts = 0
    while time.time() < deadline:
        S, S_set, counts = get_initial_greedy()
        
        # Local search
        S, S_set, counts = local_search(S, S_set, counts)
        current_size = len(S)
        
        if current_size > best_size:
            best_size = current_size
            best_set = S[:]
        
        num_restarts += 1
    
    return best_set

# Test
if __name__ == "__main__":
    res = construct()
    print(len(res))
