"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80
    
    def is_valid_addition(x, S_set, S_list):
        for i in range(len(S_list)):
            a = S_list[i]
            sa = x + a
            if sa not in S_set:
                continue
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                sb = x + b
                if sb not in S_set:
                    continue
                ab = a + b
                if ab in S_set:
                    return False
        return True
    
    # Start with all large elements
    S = list(range(65, 129))
    S_set = set(S)
    
    # Greedy: add from 64 down to 1
    for x in range(64, 0, -1):
        if time.time() > deadline:
            break
        if is_valid_addition(x, S_set, S):
            S.append(x)
            S_set.add(x)
    
    # Local search: try adding any remaining element
    S_set = set(S)
    for x in range(1, 129):
        if time.time() > deadline:
            break
        if x not in S_set and is_valid_addition(x, S_set, S):
            S.append(x)
            S_set.add(x)
    
    # Try swaps to potentially enable more additions
    for _ in range(50):  # limited iterations
        if time.time() > deadline:
            break
        S_set = set(S)
        found_swap = False
        for i in range(len(S)):
            if time.time() > deadline:
                break
            a = S[i]
            for x in range(1, 129):
                if x in S_set:
                    continue
                new_S = S[:i] + S[i+1:]
                new_S_set = S_set - {a}
                if is_valid_addition(x, new_S_set, new_S):
                    # Check if we can then add something else
                    S2 = new_S + [x]
                    S2_set = new_S_set | {x}
                    added_more = False
                    for y in range(1, 129):
                        if y in S2_set:
                            continue
                        if is_valid_addition(y, S2_set, S2):
                            S2.append(y)
                            S2_set.add(y)
                            added_more = True
                            break
                    if added_more:
                        S = S2
                        S_set = S2_set
                        found_swap = True
                        break
            if found_swap:
                break
        if not found_swap:
            break
    
    return [int(x) for x in S]

# EVOLVE-BLOCK-END
