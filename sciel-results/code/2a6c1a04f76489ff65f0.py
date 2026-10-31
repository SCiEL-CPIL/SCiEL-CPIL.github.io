import math
import time
import random

def calculate_score(U):
    if not U:
        return 0.0
    U_set = set(U)
    max_u = max(U_set)
    
    sumset = set()
    diffset = set()
    n = len(U_set)
    U_list = list(U_set)
    
    for i in range(n):
        for j in range(i, n):
            sumset.add(U_list[i] + U_list[j])
        for j in range(n):
            diffset.add(U_list[i] - U_list[j])
            
    size_sum = len(sumset)
    size_diff = len(diffset)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / size_sum
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
        
    score = 1 + math.log(ratio) / denom
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    best_score = -1
    best_set = [0]
    
    # Start with small Golomb rulers as seeds
    seed_rulers = [
        [0],
        [0, 1],
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 29, 34],
    ]
    
    def hill_climb(U, max_steps=5000):
        """Local search: try adding, removing, or modifying elements."""
        U = sorted(U)
        current_score = calculate_score(U)
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding a new element
            max_u = max(U) if U else 0
            # Search for good candidates to add
            candidates_to_try = []
            
            # Near existing elements (for clustering sums)
            for u in U:
                for delta in range(1, min(20, max_u + 1)):
                    c1 = u + delta
                    c2 = u - delta
                    if c1 >= 0 and c1 not in U:
                        candidates_to_try.append(c1)
                    if c2 >= 0 and c2 not in U:
                        candidates_to_try.append(c2)
            
            # Some larger jumps
            for delta in range(max_u // 2, max_u + 10):
                c = max_u + delta
                if c not in U:
                    candidates_to_try.append(c)
            
            # Try a random subset of candidates (limit work per step)
            random.shuffle(candidates_to_try)
            for c in candidates_to_try[:30]:
                new_U = sorted(U + [c])
                new_score = calculate_score(new_U)
                if new_score > current_score + 1e-12:
                    U = new_U
                    current_score = new_score
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try removing an element (if set is too large)
            if len(U) > 1:
                for idx in range(len(U)):
                    if time.time() > deadline:
                        break
                    new_U = U[:idx] + U[idx+1:]
                    new_score = calculate_score(new_U)
                    if new_score > current_score + 1e-12:
                        U = new_U
                        current_score = new_score
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try modifying an element (replace with nearby value)
            if len(U) > 1:
                for idx in range(1, len(U)):  # Keep 0 fixed
                    if time.time() > deadline:
                        break
                    u = U[idx]
                    for delta in range(1, min(15, u + 1)):
                        new_val = u + delta
                        if new_val not in U:
                            new_U = sorted(U[:idx] + [new_val] + U[idx+1:])
                            new_score = calculate_score(new_U)
                            if new_score > current_score + 1e-12:
                                U = new_U
                                current_score = new_score
                                improved = True
                                break
                        if new_val - delta > 0 and (u - delta) not in U and (u - delta) != 0:
                            new_val2 = u - delta
                            new_U = sorted(U[:idx] + [new_val2] + U[idx+1:])
                            new_score = calculate_score(new_U)
                            if new_score > current_score + 1e-12:
                                U = new_U
                                current_score = new_score
                                improved = True
                                break
                    if improved:
                        break
            
            if not improved:
                # Stuck - try a random perturbation
                if len(U) > 2:
                    idx = random.randint(1, len(U) - 1)
                    max_u = max(U)
                    new_val = random.randint(0, max_u + 10)
                    if new_val not in U or new_val == U[idx]:
                        new_U = sorted(U[:idx] + [new_val] + U[idx+1:])
                        if len(new_U) == len(U):
                            new_score = calculate_score(new_U)
                            if new_score > current_score - 0.01:  # Accept slight degradation
                                U = new_U
                                current_score = new_score
                else:
                    break
        
        return U, current_score
    
    # Run hill climbing from multiple seeds
    for seed in seed_rulers:
        if time.time() > deadline:
            break
        
        result_U, result_score = hill_climb(list(seed), max_steps=3000)
        
        if result_score > best_score:
            best_score = result_score
            best_set = result_U
    
    # Also try building from scratch with greedy + hill climb
    if time.time() < deadline - 10:
        U = [0]
        for _ in range(8):
            if time.time() > deadline:
                break
            max_u = max(U)
            best_candidate = None
            best_candidate_score = -1
            
            for c in range(max_u + 1, max_u + 50):
                new_U = sorted(U + [c])
                s = calculate_score(new_U)
                if s > best_candidate_score:
                    best_candidate_score = s
                    best_candidate = c
            
            if best_candidate is not None:
                U.append(best_candidate)
        
        result_U, result_score = hill_climb(U, max_steps=3000)
        if result_score > best_score:
            best_score = result_score
            best_set = result_U
    
    return best_set
