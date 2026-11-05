import time
import random

# EVOLVE-BLOCK-START

def construct():
    N = 256
    
    # Precompute 4-APs
    four_aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
            
    elem_to_4aps = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(four_aps):
        for x in ap:
            elem_to_4aps[x].append(idx)

    # Precompute 3-APs
    # A 3-AP is (a, a+d, a+2d)
    three_aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))
            
    elem_to_3aps = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(three_aps):
        for x in ap:
            elem_to_3aps[x].append(idx)

    def is_valid_addition(S, x):
        """Check if adding x to S creates a 4-AP."""
        for idx in elem_to_4aps[x]:
            ap = four_aps[idx]
            others = [y for y in ap if y != x]
            if all(y in S for y in others):
                return False
        return True

    def count_3aps(S):
        count = 0
        S_set = set(S)
        for idx in range(len(three_aps)):
            a, b, c = three_aps[idx]
            if a in S_set and b in S_set and c in S_set:
                count += 1
        return count

    # Initial greedy construction
    S = set()
    for x in range(1, N + 1):
        if is_valid_addition(S, x):
            S.add(x)
            
    best_S = set(S)
    best_score = count_3aps(best_S)
    
    start_time = time.time()
    time_limit = 55.0
    
    random.seed(42)
    
    # Local Search with Incremental Scoring
    # We maintain current_score.
    current_score = best_score
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try a batch of swaps
        for _ in range(200):
            if time.time() - start_time > time_limit:
                break
                
            # Pick u from S, v from [1,N]\S
            u = random.choice(list(S))
            v = random.randint(1, N)
            if v in S:
                continue
                
            # Check validity of swap (S - {u} + {v})
            # Valid if no 4-AP is formed by v in S-{u}
            valid_swap = True
            for idx in elem_to_4aps[v]:
                ap = four_aps[idx]
                others = [y for y in ap if y != v]
                if all(y in S for y in others):
                    # If u is not in others, then u doesn't help break this AP
                    if u not in others:
                        valid_swap = False
                        break
                    # If u is in others, this specific AP is broken by removing u
                    # But we must check if ANY other AP is formed.
                    # The loop continues.
            
            if not valid_swap:
                continue
                
            # Calculate score change
            # Remove u: subtract 3-APs containing u that are currently fully in S
            # Add v: add 3-APs containing v that will be fully in S-{u}+{v}
            
            delta = 0
            
            # Impact of removing u
            for idx in elem_to_3aps[u]:
                ap = three_aps[idx]
                others = [y for y in ap if y != u]
                if all(y in S for y in others):
                    delta -= 1
                    
            # Impact of adding v
            for idx in elem_to_3aps[v]:
                ap = three_aps[idx]
                others = [y for y in ap if y != v]
                # The others must be in S-{u}
                # So they must be in S AND not equal to u
                if all(y in S for y in others) and u not in others:
                    delta += 1
                    
            new_score = current_score + delta
            
            if new_score > current_score:
                # Accept move
                S.remove(u)
                S.add(v)
                current_score = new_score
                best_S = set(S)
                best_score = current_score
                improved = True
            elif new_score == current_score:
                # Accept with some probability to escape plateaus
                if random.random() < 0.1:
                    S.remove(u)
                    S.add(v)
                    # current_score unchanged
                    # Update best if it's a tie? No, best is already max.
                    # But we might find a better one later from this new state.
                    pass
            # else: reject

        # If stuck, we could try a random restart or just continue.
        # For now, continue.

    return sorted(best_S)

# EVOLVE-BLOCK-END
