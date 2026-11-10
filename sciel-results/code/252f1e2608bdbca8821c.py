"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 256
    max_size = 64
    
    # State: 0 = unused, 1 = in A, 2 = in B
    # We will use lists for A and B, and sets for sums and diffs.
    
    def init_greedy():
        """
        Greedy initialization to get a valid starting point.
        """
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # Try to build A and B simultaneously using a simple heuristic
        # Prioritize smaller numbers first to pack sums densely?
        # Or larger numbers?
        # Let's try random shuffles and pick the best.
        
        best_A, best_B = [], []
        best_score = 0
        
        for _ in range(10):
            if time.time() > deadline - 70:
                break
            
            order = list(range(1, n + 1))
            random.shuffle(order)
            
            A = []
            B = []
            sums_A = set()
            sums_B = set()
            diffs_A = set()
            diffs_B = set()
            
            for x in order:
                # Try to add to A
                can_add_A = True
                # Check Sidon for A
                if (x + x) in sums_A:
                    can_add_A = False
                else:
                    for a in A:
                        if (x + a) in sums_A:
                            can_add_A = False
                            break
                
                # Check Diff Disjointness for A
                if can_add_A:
                    for a in A:
                        d1 = x - a
                        d2 = a - x
                        if d1 in diffs_B or d2 in diffs_B:
                            can_add_A = False
                            break
                
                # Try to add to B
                can_add_B = True
                if (x + x) in sums_B:
                    can_add_B = False
                else:
                    for b in B:
                        if (x + b) in sums_B:
                            can_add_B = False
                            break
                
                if can_add_B:
                    for b in B:
                        d1 = x - b
                        d2 = b - x
                        if d1 in diffs_A or d2 in diffs_A:
                            can_add_B = False
                            break
                
                # Decide where to put x
                # Balance sizes
                if can_add_A and (not can_add_B or len(A) <= len(B)):
                    A.append(x)
                    sums_A.add(x + x)
                    for a in A[:-1]:
                        sums_A.add(x + a)
                    for a in A[:-1]:
                        diffs_A.add(x - a)
                        diffs_A.add(a - x)
                elif can_add_B:
                    B.append(x)
                    sums_B.add(x + x)
                    for b in B[:-1]:
                        sums_B.add(x + b)
                    for b in B[:-1]:
                        diffs_B.add(x - b)
                        diffs_B.add(b - x)
            
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_A = A
                best_B = B
                
        return best_A, best_B

    # Initial state
    A, B = init_greedy()
    
    # Build data structures for fast updates
    # We maintain:
    # A: list of elements
    # B: list of elements
    # sums_A: set of sums
    # sums_B: set of sums
    # diffs_A: set of diffs
    # diffs_B: set of diffs
    # in_A: boolean array or set for O(1) lookup
    # in_B: boolean array or set for O(1) lookup
    
    in_A = set(A)
    in_B = set(B)
    
    def build_sums(s):
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.add(s[i] + s[j])
        return sums
        
    def build_diffs(s):
        diffs = set()
        for i in range(len(s)):
            for j in range(len(s)):
                if i != j:
                    diffs.add(s[j] - s[i])
        return diffs

    sums_A = build_sums(A)
    sums_B = build_sums(B)
    diffs_A = build_diffs(A)
    diffs_B = build_diffs(B)
    
    current_score = len(A) + len(B)
    best_score = current_score
    best_A = list(A)
    best_B = list(B)
    
    # Local Search
    # Temperature schedule
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    # Helper to check if x can be added to set S with sums S_sums and diffs S_diffs,
    # and other diffs O_diffs
    def can_add(x, S, S_sums, S_diffs, O_diffs):
        # Sidon check
        if (x + x) in S_sums:
            return False
        for a in S:
            if (x + a) in S_sums:
                return False
        
        # Diff check
        for a in S:
            d1 = x - a
            d2 = a - x
            if d1 in O_diffs or d2 in O_diffs:
                return False
        
        return True

    # Helper to add x to S, updating sums and diffs
    def add_to_set(x, S, S_sums, S_diffs):
        S.append(x)
        S_sums.add(x + x)
        for a in S[:-1]:
            S_sums.add(x + a)
            S_diffs.add(x - a)
            S_diffs.add(a - x)

    # Helper to remove x from S, updating sums and diffs
    def remove_from_set(x, S, S_sums, S_diffs):
        S.remove(x)
        # Remove x+x
        S_sums.discard(x + x)
        # Remove x+a and update diffs
        for a in S:
            S_sums.discard(x + a)
            S_diffs.discard(x - a)
            S_diffs.discard(a - x)

    steps = 0
    while time.time() < deadline - 5:
        if T < T_min:
            T = T_min
            
        # Pick a random element
        x = random.randint(1, n)
        
        # Current state of x
        if x in in_A:
            current_state = 1
        elif x in in_B:
            current_state = 2
        else:
            current_state = 0
            
        # Pick a new state
        possible_states = [0, 1, 2]
        possible_states.remove(current_state)
        new_state = random.choice(possible_states)
        
        # Calculate delta score
        delta_score = 0
        if current_state == 0 and new_state in (1, 2):
            delta_score = 1
        elif current_state in (1, 2) and new_state == 0:
            delta_score = -1
        else:
            delta_score = 0
            
        # If delta_score is 0, we are just moving between A and B.
        # This might help rearrange to allow future additions.
        
        # Try to apply the move
        valid = True
        
        if new_state == 1:
            # Move x to A
            if current_state == 2:
                # Remove from B
                remove_from_set(x, B, sums_B, diffs_B)
                in_B.discard(x)
            # Add to A
            if not can_add(x, A, sums_A, diffs_A, diffs_B):
                valid = False
                # Revert if we removed from B
                if current_state == 2:
                    add_to_set(x, B, sums_B, diffs_B)
                    in_B.add(x)
            
        elif new_state == 2:
            # Move x to B
            if current_state == 1:
                # Remove from A
                remove_from_set(x, A, sums_A, diffs_A)
                in_A.discard(x)
            # Add to B
            if not can_add(x, B, sums_B, diffs_B, diffs_A):
                valid = False
                # Revert if we removed from A
                if current_state == 1:
                    add_to_set(x, A, sums_A, diffs_A)
                    in_A.add(x)
                    
        elif new_state == 0:
            # Move x to Unused
            if current_state == 1:
                remove_from_set(x, A, sums_A, diffs_A)
                in_A.discard(x)
            elif current_state == 2:
                remove_from_set(x, B, sums_B, diffs_B)
                in_B.discard(x)
            # Always valid to remove
            
        if valid:
            new_score = current_score + delta_score
            
            # Acceptance criterion
            if new_score > current_score:
                # Always accept improvement
                pass
            else:
                # Accept with probability exp(-delta/T)
                # If delta is 0, prob is 1.
                # If delta < 0, prob < 1.
                if random.random() < math.exp(-delta_score / T):
                    pass
                else:
                    # Reject: Revert the move
                    if new_state == 1:
                        # Remove from A
                        remove_from_set(x, A, sums_A, diffs_A)
                        in_A.discard(x)
                        # Add back to old state
                        if current_state == 2:
                            add_to_set(x, B, sums_B, diffs_B)
                            in_B.add(x)
                        elif current_state == 0:
                            pass
                    elif new_state == 2:
                        # Remove from B
                        remove_from_set(x, B, sums_B, diffs_B)
                        in_B.discard(x)
                        # Add back to old state
                        if current_state == 1:
                            add_to_set(x, A, sums_A, diffs_A)
                            in_A.add(x)
                        elif current_state == 0:
                            pass
                    elif new_state == 0:
                        # Add back to old state
                        if current_state == 1:
                            add_to_set(x, A, sums_A, diffs_A)
                            in_A.add(x)
                        elif current_state == 2:
                            add_to_set(x, B, sums_B, diffs_B)
                            in_B.add(x)
                    
                    # Update score back
                    # Note: current_score is not changed yet if we reject
                    # We only update current_score if we accept
                    # So we just continue
                    T *= cooling
                    steps += 1
                    continue
                    
            # If accepted, update state
            current_score = new_score
            
            if new_score > best_score:
                best_score = new_score
                best_A = list(A)
                best_B = list(B)
                
            T *= cooling
            steps += 1
            
        else:
            # Invalid move, do nothing, just cool
            T *= cooling
            steps += 1

    # Final verification and cleanup
    # Ensure best_A and best_B are valid
    def is_valid(A, B):
        sums_A = build_sums(A)
        sums_B = build_sums(B)
        diffs_A = build_diffs(A)
        diffs_B = build_diffs(B)
        
        # Check Sidon
        if len(sums_A) != len(A) * (len(A) + 1) // 2:
            return False
        if len(sums_B) != len(B) * (len(B) + 1) // 2:
            return False
            
        # Check Disjoint Diffs
        if diffs_A & diffs_B:
            return False
            
        return True

    if not is_valid(best_A, best_B):
        # Fallback to a simple valid set if something went wrong
        # This should not happen if logic is correct, but safe to have
        best_A, best_B = init_greedy()
        
    return [best_A, best_B]

# EVOLVE-BLOCK-END
