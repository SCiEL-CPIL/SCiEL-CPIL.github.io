import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 90s limit

    best_size = 0
    best_A = []
    best_B = []

    N = 256

    def try_add_to_set(x, S, sums, diffs_other, sums_other):
        """Try to add x to set S. Returns (success, new_sums, new_diffs) or (False, 0, 0)."""
        # Check sum conflicts
        temp_sums = 0
        for a in S:
            s = x + a
            if (sums >> s) & 1:
                return (False, 0, 0)
            temp_sums |= (1 << s)
        if (sums >> (2 * x)) & 1:
            return (False, 0, 0)
        temp_sums |= (1 << (2 * x))
        
        # Check diff conflicts with other set's diffs
        temp_diffs = 0
        for a in S:
            d = abs(x - a)
            if d > 0:
                if (diffs_other >> d) & 1:
                    return (False, 0, 0)
                temp_diffs |= (1 << d)
        
        return (True, temp_sums, temp_diffs)

    def greedy_pass():
        """Run one greedy pass with random ordering."""
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        A = []
        B = []
        sums_A = 0
        sums_B = 0
        diffs_A = 0
        diffs_B = 0
        
        for x in order:
            # Try adding to A
            success, new_sums, new_diffs = try_add_to_set(x, A, sums_A, diffs_B, 0)
            if success:
                A.append(x)
                sums_A |= new_sums
                diffs_A |= new_diffs
            else:
                # Try adding to B
                success2, new_sums2, new_diffs2 = try_add_to_set(x, B, sums_B, diffs_A, 0)
                if success2:
                    B.append(x)
                    sums_B |= new_sums2
                    diffs_B |= new_diffs2
        
        return A, B

    def polish(A, B, time_limit):
        """Local search polish: try insertions, swaps, and deletions."""
        A = list(A)
        B = list(B)
        polish_deadline = time.time() + time_limit
        improved = True
        
        while improved and time.time() < polish_deadline:
            improved = False
            best_gain = 0
            best_move = None
            
            # Compute current sums/diffs
            def compute_masks(S):
                sums = 0
                diffs = 0
                for i, a in enumerate(S):
                    sums |= (1 << (2 * a))
                    for j in range(i):
                        s = a + S[j]
                        sums |= (1 << s)
                        d = abs(a - S[j])
                        if d > 0:
                            diffs |= (1 << d)
                return sums, diffs
            
            sums_A, diffs_A = compute_masks(A)
            sums_B, diffs_B = compute_masks(B)
            all_elements = set(A) | set(B)
            available = [x for x in range(1, N + 1) if x not in all_elements]
            
            # Try insertions
            for x in available:
                # Try inserting into A
                success, ns, nd = try_add_to_set(x, A, sums_A, diffs_B, 0)
                if success:
                    if best_move is None or best_gain < 1:
                        best_gain = 1
                        best_move = ('insert', x, 'A')
                
                # Try inserting into B
                success2, ns2, nd2 = try_add_to_set(x, B, sums_B, diffs_A, 0)
                if success2:
                    if best_move is None or best_gain < 1:
                        best_gain = 1
                        best_move = ('insert', x, 'B')
            
            # Try swaps: remove y from A, add x to A (or vice versa)
            for x in available:
                for i, y in enumerate(A):
                    # Try replacing y with x in A
                    A_test = A[:i] + A[i+1:]
                    sums_test, diffs_test = compute_masks(A_test)
                    success, ns, nd = try_add_to_set(x, A_test, sums_test, diffs_B, 0)
                    if success:
                        if best_move is None or best_gain < 0:
                            best_gain = 0
                            best_move = ('swap', x, i, 'A')
                
                for i, y in enumerate(B):
                    B_test = B[:i] + B[i+1:]
                    sums_test, diffs_test = compute_masks(B_test)
                    success, ns, nd = try_add_to_set(x, B_test, sums_test, diffs_A, 0)
                    if success:
                        if best_move is None or best_gain < 0:
                            best_gain = 0
                            best_move = ('swap', x, i, 'B')
            
            # Try moving elements between sets
            for i, y in enumerate(A):
                A_test = A[:i] + A[i+1:]
                sums_A_test, diffs_A_test = compute_masks(A_test)
                success, ns, nd = try_add_to_set(y, B, sums_B, diffs_A_test, 0)
                if success:
                    if best_move is None or best_gain < 0:
                        best_gain = 0
                        best_move = ('move', y, i, 'A_to_B')
            
            for i, y in enumerate(B):
                B_test = B[:i] + B[i+1:]
                sums_B_test, diffs_B_test = compute_masks(B_test)
                success, ns, nd = try_add_to_set(y, A, sums_A, diffs_B_test, 0)
                if success:
                    if best_move is None or best_gain < 0:
                        best_gain = 0
                        best_move = ('move', y, i, 'B_to_A')
            
            if best_move is not None and best_gain > 0:
                improved = True
                op, x, idx, target = best_move
                if op == 'insert':
                    if target == 'A':
                        A.append(x)
                    else:
                        B.append(x)
                elif op == 'swap':
                    if target == 'A':
                        A[idx] = x
                    else:
                        B[idx] = x
                elif op == 'move':
                    if target == 'A_to_B':
                        A.pop(idx)
                        B.append(x)
                    else:
                        B.pop(idx)
                        A.append(x)
        
        return A, B

    # Main loop: greedy passes
    restart_count = 0
    while time.time() < deadline:
        A, B = greedy_pass()
        restart_count += 1
        
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = list(A)
            best_B = list(B)
    
    # Polish the best solution found
    if best_size > 0 and time.time() < deadline - 5:
        polish_time = min(10.0, max(1.0, deadline - time.time() - 5))
        polished_A, polished_B = polish(best_A, best_B, polish_time)
        polished_size = len(polished_A) + len(polished_B)
        if polished_size > best_size:
            best_size = polished_size
            best_A = polished_A
            best_B = polished_B

    # Ensure non-empty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
