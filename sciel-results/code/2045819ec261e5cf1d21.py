import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    n = 256
    max_size = 64

    def get_state(A, B):
        """Compute sum and diff sets for A and B."""
        sums_A = set()
        diffs_A = set()
        for i in range(len(A)):
            ai = A[i]
            sums_A.add(ai + ai)
            for j in range(i):
                aj = A[j]
                sums_A.add(ai + aj)
                diffs_A.add(ai - aj)
                diffs_A.add(aj - ai)
        
        sums_B = set()
        diffs_B = set()
        for i in range(len(B)):
            bi = B[i]
            sums_B.add(bi + bi)
            for j in range(i):
                bj = B[j]
                sums_B.add(bi + bj)
                diffs_B.add(bi - bj)
                diffs_B.add(bj - bi)
                
        return sums_A, diffs_A, sums_B, diffs_B

    def can_add_A(x, A, sums_A, diffs_A, diffs_B):
        """Check if x can be added to A."""
        if x + x in sums_A:
            return False
        for a in A:
            if x + a in sums_A:
                return False
            if x - a in diffs_B:
                return False
            if a - x in diffs_B:
                return False
        return True

    def can_add_B(x, B, sums_B, diffs_A, diffs_B):
        """Check if x can be added to B."""
        if x + x in sums_B:
            return False
        for b in B:
            if x + b in sums_B:
                return False
            if x - b in diffs_A:
                return False
            if b - x in diffs_A:
                return False
        return True

    def greedy_fill(order, max_iters=10000):
        A = []
        B = []
        sums_A = set()
        diffs_A = set()
        sums_B = set()
        diffs_B = set()

        for x in order:
            if len(A) >= max_size and len(B) >= max_size:
                break
            if time.time() > deadline - 2.0:
                break

            # Try A first
            if len(A) < max_size and can_add_A(x, A, sums_A, diffs_A, diffs_B):
                A.append(x)
                sums_A.add(x + x)
                for a in A[:-1]:
                    sums_A.add(x + a)
                    diffs_A.add(x - a)
                    diffs_A.add(a - x)
                continue

            # Try B
            if len(B) < max_size and can_add_B(x, B, sums_B, diffs_A, diffs_B):
                B.append(x)
                sums_B.add(x + x)
                for b in B[:-1]:
                    sums_B.add(x + b)
                    diffs_B.add(x - b)
                    diffs_B.add(b - x)
                continue

        return A, B, sums_A, diffs_A, sums_B, diffs_B

    def local_search(A, B, sums_A, diffs_A, sums_B, diffs_B):
        """
        Iteratively try to add elements or swap elements to improve size.
        """
        in_AB = set(A) | set(B)
        
        while time.time() < deadline - 3.0:
            added = False
            swapped = False
            
            # 1. Try to add any new element
            candidates = [x for x in range(1, n + 1) if x not in in_AB]
            random.shuffle(candidates)
            
            for x in candidates:
                if len(A) < max_size and can_add_A(x, A, sums_A, diffs_A, diffs_B):
                    A.append(x)
                    in_AB.add(x)
                    sums_A.add(x + x)
                    for a in A[:-1]:
                        sums_A.add(x + a)
                        diffs_A.add(x - a)
                        diffs_A.add(a - x)
                    added = True
                    break
                elif len(B) < max_size and can_add_B(x, B, sums_B, diffs_A, diffs_B):
                    B.append(x)
                    in_AB.add(x)
                    sums_B.add(x + x)
                    for b in B[:-1]:
                        sums_B.add(x + b)
                        diffs_B.add(x - b)
                        diffs_B.add(b - x)
                    added = True
                    break
            
            if added:
                continue

            # 2. If no add possible, try a swap (kick)
            # Pick a random element from A or B to remove, then try to refill
            if A:
                idx_a = random.randint(0, len(A) - 1)
                a_val = A[idx_a]
                # Remove a_val from A
                A.pop(idx_a)
                # Recompute state for A (expensive but safe)
                sums_A_new, diffs_A_new, sums_B_new, diffs_B_new = get_state(A, B)
                
                # Try to add any candidate to A or B with the new state
                # We prioritize adding a_val back to B if possible, or a new x to A
                # But let's just try to add ANY x to A or B to see if we can grow or maintain size
                # Actually, we want to see if removing a_val allows a *different* x to be added to A
                # or if a_val can be moved to B.
                
                # Try to add a_val to B
                if len(B) < max_size and can_add_B(a_val, B, sums_B_new, diffs_A_new, diffs_B_new):
                    B.append(a_val)
                    in_AB.add(a_val) # It was already in in_AB, but logic holds
                    # Update state
                    sums_B_new.add(a_val + a_val)
                    for b in B[:-1]:
                        sums_B_new.add(a_val + b)
                        diffs_B_new.add(a_val - b)
                        diffs_B_new.add(b - a_val)
                    # State is updated. A is smaller, B is larger. Total size same.
                    # But maybe this unblocks something?
                    # Let's just accept this move and continue loop.
                    A, B = A, B
                    sums_A, diffs_A, sums_B, diffs_B = sums_A_new, diffs_A_new, sums_B_new, diffs_B_new
                    swapped = True
                else:
                    # Try to add a new x to A (now that a_val is gone)
                    # We check all x not in A or B
                    # To save time, we just pick a few random ones
                    # If we find one, we replace a_val with x. Size same.
                    # If we find one that allows *further* growth, that's great.
                    # But for now, let's just try to add a_val back to A or a new x to A.
                    # If we can't add anything, we revert.
                    
                    # Try to add a_val back to A (revert)
                    if can_add_A(a_val, A, sums_A_new, diffs_A_new, diffs_B_new):
                        A.append(a_val)
                        # Restore state
                        sums_A = sums_A_new
                        diffs_A = diffs_A_new
                        sums_B = sums_B_new
                        diffs_B = diffs_B_new
                        # Update with a_val
                        sums_A.add(a_val + a_val)
                        for a in A[:-1]:
                            sums_A.add(a_val + a)
                            diffs_A.add(a_val - a)
                            diffs_A.add(a - a_val)
                        # No change in size, just a shuffle.
                    else:
                        # Try to add a random new x to A
                        found = False
                        for _ in range(10): # Try 10 random candidates
                            if not candidates: break
                            x = random.choice(candidates)
                            if x in in_AB: continue
                            if len(A) < max_size and can_add_A(x, A, sums_A_new, diffs_A_new, diffs_B_new):
                                A.append(x)
                                in_AB.add(x)
                                sums_A_new.add(x + x)
                                for a in A[:-1]:
                                    sums_A_new.add(x + a)
                                    diffs_A_new.add(x - a)
                                    diffs_A_new.add(a - x)
                                A, B = A, B
                                sums_A, diffs_A, sums_B, diffs_B = sums_A_new, diffs_A_new, sums_B_new, diffs_B_new
                                found = True
                                swapped = True
                                break
                        if not found:
                            # Revert: put a_val back in A
                            A.append(a_val)
                            sums_A = sums_A_new
                            diffs_A = diffs_A_new
                            sums_B = sums_B_new
                            diffs_B = diffs_B_new
                            sums_A.add(a_val + a_val)
                            for a in A[:-1]:
                                sums_A.add(a_val + a)
                                diffs_A.add(a_val - a)
                                diffs_A.add(a - a_val)
            elif B:
                # Similar logic for B, but less likely to be needed if A is empty
                pass
            else:
                break

            if not added and not swapped:
                # Stuck? Try a more aggressive kick: remove 2 elements?
                # Or just break if no progress for a while?
                # For now, break to allow a new restart if this is part of a loop.
                # But local_search is called once per restart.
                # Let's add a counter to avoid infinite loops on same state.
                # Actually, the random shuffle of candidates and random pick of a_val ensures progress in state space.
                pass

        return A, B, sums_A, diffs_A, sums_B, diffs_B

    def validate(A, B):
        if not A or not B:
            return False
        if len(A) > max_size or len(B) > max_size:
            return False

        sums_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums_A:
                    return False
                sums_A.add(s)

        sums_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                s = B[i] + B[j]
                if s in sums_B:
                    return False
                sums_B.add(s)

        diffs_A = set()
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    diffs_A.add(A[j] - A[i])

        diffs_B = set()
        for i in range(len(B)):
            for j in range(len(B)):
                if i != j:
                    diffs_B.add(B[j] - B[i])

        if diffs_A & diffs_B:
            return False

        return True

    best_A, best_B = [], []
    best_score = 0

    # Multiple restarts
    while time.time() < deadline - 5.0:
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        A, B, sums_A, diffs_A, sums_B, diffs_B = greedy_fill(order)
        
        # Local search to improve
        A, B, sums_A, diffs_A, sums_B, diffs_B = local_search(A, B, sums_A, diffs_A, sums_B, diffs_B)
        
        if validate(A, B):
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_A, best_B = A[:], B[:]
                
        # If we hit a high score, we can be more aggressive with time or break
        if best_score >= 35: # Heuristic threshold
             # Try a few more with less time each?
             # Or just break if we have very little time left
             if time.time() > deadline - 10.0:
                 break

    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]

    return [best_A, best_B]
