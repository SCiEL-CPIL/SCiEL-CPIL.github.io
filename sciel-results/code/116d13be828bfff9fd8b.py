"""Two Sidon sets with disjoint difference sets – Bitset-based fast search.
Uses integer bitmasks for sum and difference collisions to enable high-throughput
local search within the time budget."""

import time
import random

N = 256
MAX_SIZE = 64
DEADLINE = 55.0  # Safety margin before 60s limit

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    
    # State: lists of elements, and bitmasks for sums and diffs
    # Max sum is 256+256=512. Max diff is 255.
    # We use Python ints as bitsets.
    
    A = []
    B = []
    sums_A = 0
    sums_B = 0
    diffs_A = 0
    diffs_B = 0
    
    def can_add_A(x):
        # Check if x can be added to A
        # 1. Check x+x against sums_A
        if (sums_A >> (x + x)) & 1:
            return False
        # 2. Check x+a against sums_A and diffs against diffs_A and diffs_B
        for a in A:
            s = x + a
            d = abs(x - a)
            if (sums_A >> s) & 1:
                return False
            if (diffs_A >> d) & 1:
                return False
            if (diffs_B >> d) & 1:
                return False
        return True

    def add_A(x):
        nonlocal sums_A, diffs_A
        sums_A |= (1 << (x + x))
        for a in A:
            s = x + a
            d = abs(x - a)
            sums_A |= (1 << s)
            diffs_A |= (1 << d)
        A.append(x)

    def can_add_B(x):
        # Check if x can be added to B
        if (sums_B >> (x + x)) & 1:
            return False
        for b in B:
            s = x + b
            d = abs(x - b)
            if (sums_B >> s) & 1:
                return False
            if (diffs_B >> d) & 1:
                return False
            if (diffs_A >> d) & 1:
                return False
        return True

    def add_B(x):
        nonlocal sums_B, diffs_B
        sums_B |= (1 << (x + x))
        for b in B:
            s = x + b
            d = abs(x - b)
            sums_B |= (1 << s)
            diffs_B |= (1 << d)
        B.append(x)

    def remove_A(x):
        nonlocal sums_A, diffs_A
        sums_A &= ~(1 << (x + x))
        for a in A:
            if a == x: continue
            s = x + a
            d = abs(x - a)
            sums_A &= ~(1 << s)
            diffs_A &= ~(1 << d)
        A.remove(x)

    def remove_B(x):
        nonlocal sums_B, diffs_B
        sums_B &= ~(1 << (x + x))
        for b in B:
            if b == x: continue
            s = x + b
            d = abs(x - b)
            sums_B &= ~(1 << s)
            diffs_B &= ~(1 << d)
        B.remove(x)

    # Phase 1: Greedy
    for x in range(1, N + 1):
        if time.time() > deadline - 5.0:
            break
        if len(A) < MAX_SIZE and can_add_A(x):
            add_A(x)
        elif len(B) < MAX_SIZE and can_add_B(x):
            add_B(x)
    
    # Initialize contribution tables after greedy (they were built incrementally)
    # They should already be correct from add_A/add_B calls

    # Phase 2: Local Search with multiple strategies and restarts
    best_score = len(A) + len(B)
    best_A = list(A)
    best_B = list(B)
    
    # Track which elements are in A or B for fast candidate lookup
    in_A = set(A)
    in_B = set(B)
    
    def update_in_sets():
        nonlocal in_A, in_B
        in_A = set(A)
        in_B = set(B)
    
    def get_candidates():
        """Return elements not in A or B."""
        in_sets = in_A | in_B
        return [x for x in range(1, N + 1) if x not in in_sets]
    
    def try_add_to_best(x, target='A'):
        """Try to add x to target set, update best if improved."""
        nonlocal best_score, best_A, best_B
        if target == 'A' and len(A) < MAX_SIZE and can_add_A(x):
            add_A(x)
            in_A.add(x)
            if len(A) + len(B) > best_score:
                best_score = len(A) + len(B)
                best_A = list(A)
                best_B = list(B)
            return True
        elif target == 'B' and len(B) < MAX_SIZE and can_add_B(x):
            add_B(x)
            in_B.add(x)
            if len(A) + len(B) > best_score:
                best_score = len(A) + len(B)
                best_A = list(A)
                best_B = list(B)
            return True
        return False
    
    # Multiple restarts with different random seeds
    for restart in range(15):
        if time.time() > deadline - 3.0:
            break
            
        rng = random.Random(1000 + restart * 137)
        
        # Reset state to best found so far
        A = list(best_A)
        B = list(best_B)
        sums_A = 0
        sums_B = 0
        diffs_A = 0
        diffs_B = 0
        contrib_sums_A = [0] * (N + 1)
        contrib_diffs_A = [0] * (N + 1)
        contrib_sums_B = [0] * (N + 1)
        contrib_diffs_B = [0] * (N + 1)
        for i, a in enumerate(A):
            sums_A |= (1 << (a + a))
            for j, a2 in enumerate(A):
                if j > i:
                    s = a + a2
                    d = abs(a - a2)
                    sums_A |= (1 << s)
                    diffs_A |= (1 << d)
                    contrib_sums_A[a] |= (1 << s)
                    contrib_diffs_A[a] |= (1 << d)
                    contrib_sums_A[a2] |= (1 << s)
                    contrib_diffs_A[a2] |= (1 << d)
        for i, b in enumerate(B):
            sums_B |= (1 << (b + b))
            for j, b2 in enumerate(B):
                if j > i:
                    s = b + b2
                    d = abs(b - b2)
                    sums_B |= (1 << s)
                    diffs_B |= (1 << d)
                    contrib_sums_B[b] |= (1 << s)
                    contrib_diffs_B[b] |= (1 << d)
                    contrib_sums_B[b2] |= (1 << s)
                    contrib_diffs_B[b2] |= (1 << d)
        
        in_A = set(A)
        in_B = set(B)
        
        # Local search iterations
        stall_count = 0
        max_stall = 150
        
        while time.time() < deadline - 3.0:
            if stall_count > max_stall:
                break
                
            candidates = get_candidates()
            
            # Strategy 1: Try to add random elements
            if candidates and (len(A) < MAX_SIZE or len(B) < MAX_SIZE):
                for _ in range(16):
                    x = rng.choice(candidates)
                    if len(A) < MAX_SIZE and rng.random() < 0.5:
                        if try_add_to_best(x, 'A'):
                            stall_count = 0
                    elif len(B) < MAX_SIZE:
                        if try_add_to_best(x, 'B'):
                            stall_count = 0
            
            # Strategy 2: 1-for-2 swap in A
            if len(A) >= 2:
                idx1 = rng.randint(0, len(A) - 1)
                idx2 = rng.randint(0, len(A) - 2)
                if idx2 >= idx1:
                    idx2 += 1
                a1 = A[idx1]
                a2 = A[idx2]
                remove_A(a1)
                in_A.discard(a1)
                remove_A(a2)
                in_A.discard(a2)
                added = 0
                added_elements = []
                for x in candidates:
                    if added >= 3:
                        break
                    if len(A) < MAX_SIZE and can_add_A(x):
                        add_A(x)
                        in_A.add(x)
                        added_elements.append(x)
                        added += 1
                if added > 2:
                    if len(A) + len(B) > best_score:
                        best_score = len(A) + len(B)
                        best_A = list(A)
                        best_B = list(B)
                    stall_count = 0
                else:
                    # Restore: remove new additions, then add back old ones
                    for x in added_elements:
                        remove_A(x)
                        in_A.discard(x)
                    add_A(a1)
                    in_A.add(a1)
                    add_A(a2)
                    in_A.add(a2)
                    stall_count += 1
            
            # Strategy 3: 1-for-2 swap in B
            if len(B) >= 2:
                idx1 = rng.randint(0, len(B) - 1)
                idx2 = rng.randint(0, len(B) - 2)
                if idx2 >= idx1:
                    idx2 += 1
                b1 = B[idx1]
                b2 = B[idx2]
                remove_B(b1)
                in_B.discard(b1)
                remove_B(b2)
                in_B.discard(b2)
                added = 0
                added_elements = []
                for x in candidates:
                    if added >= 3:
                        break
                    if len(B) < MAX_SIZE and can_add_B(x):
                        add_B(x)
                        in_B.add(x)
                        added_elements.append(x)
                        added += 1
                if added > 2:
                    if len(A) + len(B) > best_score:
                        best_score = len(A) + len(B)
                        best_A = list(A)
                        best_B = list(B)
                    stall_count = 0
                else:
                    # Restore: remove new additions, then add back old ones
                    for x in added_elements:
                        remove_B(x)
                        in_B.discard(x)
                    add_B(b1)
                    in_B.add(b1)
                    add_B(b2)
                    in_B.add(b2)
                    stall_count += 1
            
            # Strategy 4: Cross swap
            if A and B:
                a_old = A[rng.randint(0, len(A) - 1)]
                b_old = B[rng.randint(0, len(B) - 1)]
                remove_A(a_old)
                in_A.discard(a_old)
                remove_B(b_old)
                in_B.discard(b_old)
                added_a = 0
                added_b = 0
                added_a_elements = []
                added_b_elements = []
                for x in candidates:
                    if added_a >= 2 and added_b >= 1:
                        break
                    if added_a < 2 and len(A) < MAX_SIZE and can_add_A(x):
                        add_A(x)
                        in_A.add(x)
                        added_a_elements.append(x)
                        added_a += 1
                    if added_b < 1 and len(B) < MAX_SIZE and can_add_B(x):
                        add_B(x)
                        in_B.add(x)
                        added_b_elements.append(x)
                        added_b += 1
                if added_a + added_b > 2:
                    if len(A) + len(B) > best_score:
                        best_score = len(A) + len(B)
                        best_A = list(A)
                        best_B = list(B)
                    stall_count = 0
                else:
                    # Restore: remove new additions, then add back old ones
                    for x in added_a_elements:
                        remove_A(x)
                        in_A.discard(x)
                    for x in added_b_elements:
                        remove_B(x)
                        in_B.discard(x)
                    add_A(a_old)
                    in_A.add(a_old)
                    add_B(b_old)
                    in_B.add(b_old)
                    stall_count += 1
            
            # Strategy 5: Single element removal and re-addition
            if A and len(A) > 1:
                a_old = A[rng.randint(0, len(A) - 1)]
                remove_A(a_old)
                in_A.discard(a_old)
                added = 0
                added_element = None
                for x in candidates:
                    if len(A) < MAX_SIZE and can_add_A(x):
                        add_A(x)
                        in_A.add(x)
                        added_element = x
                        added += 1
                        break
                if added > 0:
                    if len(A) + len(B) > best_score:
                        best_score = len(A) + len(B)
                        best_A = list(A)
                        best_B = list(B)
                    stall_count = 0
                else:
                    if added_element is not None:
                        remove_A(added_element)
                        in_A.discard(added_element)
                    add_A(a_old)
                    in_A.add(a_old)
                    stall_count += 1
            
            if B and len(B) > 1:
                b_old = B[rng.randint(0, len(B) - 1)]
                remove_B(b_old)
                in_B.discard(b_old)
                added = 0
                added_element = None
                for x in candidates:
                    if len(B) < MAX_SIZE and can_add_B(x):
                        add_B(x)
                        in_B.add(x)
                        added_element = x
                        added += 1
                        break
                if added > 0:
                    if len(A) + len(B) > best_score:
                        best_score = len(A) + len(B)
                        best_A = list(A)
                        best_B = list(B)
                    stall_count = 0
                else:
                    if added_element is not None:
                        remove_B(added_element)
                        in_B.discard(added_element)
                    add_B(b_old)
                    in_B.add(b_old)
                    stall_count += 1

    # Restore best
    A = list(best_A)
    B = list(best_B)
    
    # Final polish: rebuild state and do a systematic scan
    sums_A = 0
    sums_B = 0
    diffs_A = 0
    diffs_B = 0
    contrib_sums_A = [0] * (N + 1)
    contrib_diffs_A = [0] * (N + 1)
    contrib_sums_B = [0] * (N + 1)
    contrib_diffs_B = [0] * (N + 1)
    for i, a in enumerate(A):
        sums_A |= (1 << (a + a))
        for j, a2 in enumerate(A):
            if j > i:
                s = a + a2
                d = abs(a - a2)
                sums_A |= (1 << s)
                diffs_A |= (1 << d)
                contrib_sums_A[a] |= (1 << s)
                contrib_diffs_A[a] |= (1 << d)
                contrib_sums_A[a2] |= (1 << s)
                contrib_diffs_A[a2] |= (1 << d)
    for i, b in enumerate(B):
        sums_B |= (1 << (b + b))
        for j, b2 in enumerate(B):
            if j > i:
                s = b + b2
                d = abs(b - b2)
                sums_B |= (1 << s)
                diffs_B |= (1 << d)
                contrib_sums_B[b] |= (1 << s)
                contrib_diffs_B[b] |= (1 << d)
                contrib_sums_B[b2] |= (1 << s)
                contrib_diffs_B[b2] |= (1 << d)
    
    # Systematic scan for any addable elements
    in_A = set(A)
    in_B = set(B)
    for x in range(1, N + 1):
        if x in in_A or x in in_B:
            continue
        if len(A) < MAX_SIZE and can_add_A(x):
            add_A(x)
            in_A.add(x)
        elif len(B) < MAX_SIZE and can_add_B(x):
            add_B(x)
            in_B.add(x)
    
    if not A: A = [1]
    if not B: B = [2]
    
    return [A, B]
