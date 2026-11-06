import time
import random
import sys

def construct():
    N = 256
    MAX_SIZE = 64
    DEADLINE = 50.0  # Safety margin for 60s limit
    
    # Bitmasks for sums (indices 0..512) and diffs (indices 0..256)
    # We use Python integers which are arbitrary precision, acting as bitsets.
    
    A = []
    B = []
    sumA = 0
    sumB = 0
    diffA = 0
    diffB = 0
    
    # Helper to check if adding x to side 'side' is valid
    # side: 0 for A, 1 for B
    def can_add(x, side):
        if side == 0:
            elems = A
            s_mask = sumA
            d_mask = diffA
            other_d_mask = diffB
        else:
            elems = B
            s_mask = sumB
            d_mask = diffB
            other_d_mask = diffA
        
        # Check sum x+x
        s_xx = x + x
        if (s_mask >> s_xx) & 1:
            return False
        
        # Check diffs |x-a| and sums x+a
        for a in elems:
            s = x + a
            if (s_mask >> s) & 1:
                return False
            
            d = abs(x - a)
            if (d_mask >> d) & 1:
                return False
            if (other_d_mask >> d) & 1:
                return False
        return True

    def add(x, side):
        if side == 0:
            A.append(x)
            # Update sumA
            s_xx = x + x
            sumA |= (1 << s_xx)
            for a in A:
                if a == x: continue # already handled x+x? No, combinations_with_replacement includes (x,x) once.
                # My loop in can_add iterates all a. 
                # In add, I need to set bits for all pairs involving x.
                # Pairs: (x,x) and (x,a) for a in A, a != x.
                # Since A is a list, and we just appended x, we iterate previous elements.
                pass 
            # Let's just recalculate or update carefully.
            # Actually, it's safer to update incrementally.
            # s_xx is set.
            # For each a in A before x was added:
            #   sum x+a
            #   diff |x-a|
            # Since we are inside add, A already has x at the end.
            # Let's re-iterate A excluding the last one?
            # Better: pass the list of existing elements to add, or update before append.
            # Let's change add to take the element and the current list state.
            pass
        # This inline logic is getting messy. Let's use a cleaner state update.
        
    # Let's restart the internal logic for clarity and correctness.
    
    A = []
    B = []
    sumA = 0
    sumB = 0
    diffA = 0
    diffB = 0
    
    def _add_elem(x, side):
        nonlocal sumA, sumB, diffA, diffB
        if side == 0:
            # Update masks for A
            s_xx = x + x
            sumA |= (1 << s_xx)
            for a in A:
                s = x + a
                sumA |= (1 << s)
                d = abs(x - a)
                diffA |= (1 << d)
            A.append(x)
        else:
            s_xx = x + x
            sumB |= (1 << s_xx)
            for a in B:
                s = x + a
                sumB |= (1 << s)
                d = abs(x - a)
                diffB |= (1 << d)
            B.append(x)

    def _remove_elem(x, side):
        nonlocal sumA, sumB, diffA, diffB
        if side == 0:
            # We need to clear bits that are ONLY contributed by x.
            # This is hard with simple OR bitmasks if other pairs produce same sum/diff.
            # BUT, Sidon property ensures sums are unique. 
            # So sum x+x is unique to pair (x,x).
            # Sum x+a is unique to pair {x,a}.
            # Diff |x-a| is NOT necessarily unique in a Sidon set? 
            # Sidon set: pair sums unique. 
            # Differences: In a Sidon set, differences can repeat? 
            # Example: {1, 2, 4}. Sums: 2,3,4,5,6,8. Unique.
            # Diffs: 1, 2, 3, 2, 1, 0. 
            # Non-zero diffs: 1, 2, 3. 
            # Wait, 2-1=1, 4-2=2, 4-1=3. 
            # If we remove 2: A={1,4}. Diffs: 3.
            # The diff 1 (from 2-1) is gone. Diff 2 (from 4-2) is gone. Diff 3 (from 4-1) remains.
            # So we can simply re-calculate masks for the remaining set? 
            # Or, since sums are unique, we can clear specific sum bits.
            # For diffs, we must be careful. 
            # Given N=256 and MaxSize=64, re-calculating masks for the set (O(N^2)) is fast enough (64^2 = 4096 ops).
            # Local search steps might be 100k. 100k * 4096 = 400M ops. Might be too slow in Python.
            
            # Alternative: Store counts? 
            # Or, just re-calculate from scratch for the specific set when removing.
            # Let's try re-calculation. It's safer.
            
            A.remove(x)
            # Recalculate sumA and diffA
            sumA = 0
            diffA = 0
            for i in range(len(A)):
                for j in range(i, len(A)):
                    s = A[i] + A[j]
                    sumA |= (1 << s)
                    if i != j:
                        d = abs(A[i] - A[j])
                        diffA |= (1 << d)
        else:
            B.remove(x)
            sumB = 0
            diffB = 0
            for i in range(len(B)):
                for j in range(i, len(B)):
                    s = B[i] + B[j]
                    sumB |= (1 << s)
                    if i != j:
                        d = abs(B[i] - B[j])
                        diffB |= (1 << d)

    def _can_add(x, side):
        if side == 0:
            elems = A
            s_mask = sumA
            d_mask = diffA
            other_d_mask = diffB
        else:
            elems = B
            s_mask = sumB
            d_mask = diffB
            other_d_mask = diffA
            
        s_xx = x + x
        if (s_mask >> s_xx) & 1:
            return False
        
        for a in elems:
            s = x + a
            if (s_mask >> s) & 1:
                return False
            d = abs(x - a)
            if (d_mask >> d) & 1:
                return False
            if (other_d_mask >> d) & 1:
                return False
        return True

    # Initial Greedy
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    
    for x in candidates:
        # Try to add to A
        if len(A) < MAX_SIZE and _can_add(x, 0):
            _add_elem(x, 0)
        # Try to add to B
        elif len(B) < MAX_SIZE and _can_add(x, 1):
            _add_elem(x, 1)
            
    # Local Search
    start_time = time.time()
    best_size = len(A) + len(B)
    best_A = A.copy()
    best_B = B.copy()
    
    while time.time() - start_time < DEADLINE:
        # Randomly choose a side to modify
        side = random.randint(0, 1)
        elems = A if side == 0 else B
        
        if len(elems) < MAX_SIZE:
            # Try to add a random element
            x = random.randint(1, N)
            if _can_add(x, side):
                _add_elem(x, side)
                if len(A) + len(B) > best_size:
                    best_size = len(A) + len(B)
                    best_A = A.copy()
                    best_B = B.copy()
        else:
            # Try a swap: remove one, add another
            if not elems: continue
            idx = random.randint(0, len(elems) - 1)
            old_elem = elems[idx]
            
            # Remove old_elem
            _remove_elem(old_elem, side)
            
            # Try to add a new element
            x = random.randint(1, N)
            if _can_add(x, side):
                _add_elem(x, side)
                if len(A) + len(B) > best_size:
                    best_size = len(A) + len(B)
                    best_A = A.copy()
                    best_B = B.copy()
            else:
                # Revert
                _add_elem(old_elem, side)
                # Note: _add_elem assumes it's valid. We know it was valid before removal.
                # However, _add_elem updates masks. 
                # When we removed, we recalculated masks.
                # When we re-add, we use the incremental update in _add_elem?
                # My _add_elem uses incremental update (OR with new bits).
                # But _remove_elem recalculated from scratch.
                # So the state is consistent.
                # But wait, _add_elem does NOT check validity. It just adds.
                # We must ensure we don't add an invalid element.
                # Since we reverted, the state should be back to valid.
                # The incremental update in _add_elem is correct IF the current masks are correct.
                # After _remove_elem, masks are correct for the reduced set.
                # Adding old_elem back should restore the original masks.
                # So this is safe.

    return [best_A, best_B]
