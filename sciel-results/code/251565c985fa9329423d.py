import random
import time
from itertools import permutations

def construct(n):
    if n == 1:
        return "1"
    
    start_time = time.time()
    deadline = 145.0  # 5s margin
    
    word = [1]
    
    for m in range(2, n + 1):
        if time.time() > deadline - 15:
            # Fallback: simple append without optimization
            # This is just to ensure we return something
            break
            
        perms = list(permutations(range(1, m)))
        k = len(perms)
        
        # Precompute overlap matrix
        # L(p, q) is max o such that p[-o:] == q[:o]
        overlap = [[0] * k for _ in range(k)]
        for i in range(k):
            p = perms[i]
            p_str = "".join(map(str, p))
            for j in range(k):
                q = perms[j]
                q_str = "".join(map(str, q))
                # Use string slicing for speed
                max_o = min(len(p), len(q))
                for o in range(max_o, -1, -1):
                    if o == 0:
                        overlap[i][j] = 0
                        break
                    if p_str[len(p_str)-o:] == q_str[:o]:
                        overlap[i][j] = o
                        break
        
        # Optimize order
        if k > 50:
            # Greedy start
            visited = [False] * k
            path = [0]
            visited[0] = True
            for _ in range(k - 1):
                last = path[-1]
                best_next = -1
                best_score = -1
                for j in range(k):
                    if not visited[j]:
                        score = overlap[last][j]
                        if score > best_score:
                            best_score = score
                            best_next = j
                path.append(best_next)
                visited[best_next] = True
            
            best_path = path[:]
            curr_score = sum(overlap[path[i]][path[i+1]] for i in range(k-1))
            best_score_val = curr_score
            
            temp = 1.0
            cooling = 0.9995
            min_temp = 0.01
            max_iters = 200000
            
            current_path = best_path[:]
            current_score = curr_score
            
            iter_count = 0
            while temp > min_temp and iter_count < max_iters:
                if time.time() > deadline - 5:
                    break
                i = random.randint(0, k - 2)
                j = random.randint(i + 1, k - 1)
                
                # Delta calculation
                old_score = 0
                if i > 0: old_score += overlap[current_path[i-1]][current_path[i]]
                if i < k-1: old_score += overlap[current_path[i]][current_path[i+1]]
                if j > 0: old_score += overlap[current_path[j-1]][current_path[j]]
                if j < k-1: old_score += overlap[current_path[j]][current_path[j+1]]
                
                current_path[i], current_path[j] = current_path[j], current_path[i]
                
                new_score = 0
                if i > 0: new_score += overlap[current_path[i-1]][current_path[i]]
                if i < k-1: new_score += overlap[current_path[i]][current_path[i+1]]
                if j > 0: new_score += overlap[current_path[j-1]][current_path[j]]
                if j < k-1: new_score += overlap[current_path[j]][current_path[j+1]]
                
                delta = new_score - old_score
                
                if delta > 0 or random.random() < pow(2.718281828, delta / temp):
                    current_score += delta
                    if current_score > best_score_val:
                        best_score_val = current_score
                        best_path = current_path[:]
                else:
                    current_path[i], current_path[j] = current_path[j], current_path[i]
                
                temp *= cooling
                iter_count += 1
            
            path = best_path
        else:
            # Greedy for small k
            visited = [False] * k
            path = [0]
            visited[0] = True
            for _ in range(k - 1):
                last = path[-1]
                best_next = -1
                best_score = -1
                for j in range(k):
                    if not visited[j]:
                        score = overlap[last][j]
                        if score > best_score:
                            best_score = score
                            best_next = j
                path.append(best_next)
                visited[best_next] = True

        # Construct word
        new_word = []
        for idx in path:
            p = perms[idx]
            block = list(p) + [m] + list(p)
            
            # Overlap with new_word
            max_o = min(len(new_word), len(block))
            o = 0
            # Convert to string for faster comparison?
            # new_word is list of ints.
            for i in range(max_o, -1, -1):
                if new_word[len(new_word)-i:] == block[:i]:
                    o = i
                    break
            
            new_word += block[o:]
        
        word = new_word
        
    return "".join(map(str, word))
