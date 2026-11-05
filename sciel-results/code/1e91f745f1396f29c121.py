import random
import time
def construct():
    start_time = time.time()
    best_set = None
    best_score = -1
    max_val_range = 512
    
    def calc_score(S):
        S = set(S)
        max_val = max(S)
        min_val = min(S)
        max_d = (max_val - min_val) // 2
        counts = {}
        for d in range(1, max_d + 1):
            count = 0
            for a in S:
                if a + d in S and a + 2 * d in S:
                    count += 1
            if count > 0:
                counts[d] = count
        return counts, len(counts)

    def try_move(S, counts, score):
        # Returns new_set, new_counts, new_score if improved, else None
        # Pick a random element to remove
        to_remove = random.choice(list(S))
        # Try a few random additions
        for _ in range(10):
            candidate_add = random.randint(1, 512)
            if candidate_add in S:
                continue
            
            # Simulate removal of to_remove
            S_temp = S.copy()
            S_temp.remove(to_remove)
            
            # Update counts for removal
            max_d = (max(S_temp) - min(S_temp)) // 2 if S_temp else 0
            # We need to check d up to the global max possible?
            # The max d in S_temp might be smaller than before.
            # But we can just iterate up to the previous max_d or a fixed limit like 256.
            # Let's use a fixed limit of 256 for simplicity in update.
            
            new_counts = counts.copy()
            
            # Remove contributions of to_remove
            for d in range(1, 257):
                # Check if to_remove was part of an AP with diff d
                # 1. to_remove is a
                if to_remove + d in S_temp and to_remove + 2 * d in S_temp:
                    if d in new_counts:
                        new_counts[d] -= 1
                        if new_counts[d] == 0:
                            del new_counts[d]
                # 2. to_remove is a+d
                if to_remove - d in S_temp and to_remove + d in S_temp:
                    if d in new_counts:
                        new_counts[d] -= 1
                        if new_counts[d] == 0:
                            del new_counts[d]
                # 3. to_remove is a+2d
                if to_remove - 2 * d in S_temp and to_remove - d in S_temp:
                    if d in new_counts:
                        new_counts[d] -= 1
                        if new_counts[d] == 0:
                            del new_counts[d]
            
            # Add contributions of candidate_add
            S_temp.add(candidate_add)
            
            for d in range(1, 257):
                # Check if candidate_add forms an AP with diff d
                # 1. candidate_add is a
                if candidate_add + d in S_temp and candidate_add + 2 * d in S_temp:
                    new_counts[d] = new_counts.get(d, 0) + 1
                # 2. candidate_add is a+d
                if candidate_add - d in S_temp and candidate_add + d in S_temp:
                    new_counts[d] = new_counts.get(d, 0) + 1
                # 3. candidate_add is a+2d
                if candidate_add - 2 * d in S_temp and candidate_add - d in S_temp:
                    new_counts[d] = new_counts.get(d, 0) + 1
            
            new_score = len(new_counts)
            if new_score > score:
                return S_temp, new_counts, new_score
        return None

    # Initial sets
    inits = [
        list(range(1, 33)),
        random.sample(range(1, 513), 32),
        [1 + i * 16 for i in range(32)]
    ]
    
    for init in inits:
        if time.time() - start_time > 145: break
        S = set(init)
        counts, score = calc_score(S)
        
        if score > best_score:
            best_score = score
            best_set = set(S)
            
        no_improve = 0
        while time.time() - start_time < 145:
            res = try_move(S, counts, score)
            if res:
                S, counts, score = res
                no_improve = 0
                if score > best_score:
                    best_score = score
                    best_set = set(S)
            else:
                no_improve += 1
                if no_improve > 50:
                    break
                    
    return list(best_set)
