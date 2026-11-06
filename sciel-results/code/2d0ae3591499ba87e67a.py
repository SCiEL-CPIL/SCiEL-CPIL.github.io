import time
import random
import sys

N = 256
MAX_EACH = 64
DEADLINE = 55.0  # Safety margin for 60s limit

def construct():
    start_time = time.time()
    best_size = 0
    best_A = []
    best_B = []
    
    def is_valid_add(x, elems, sum_bits, diff_bits, other_diff_bits):
        """Check if x can be added to the set."""
        if (sum_bits >> (x + x)) & 1:
            return False
        for a in elems:
            s = x + a
            if (sum_bits >> s) & 1:
                return False
            d = abs(x - a)
            if (diff_bits >> d) & 1:
                return False
            if (other_diff_bits >> d) & 1:
                return False
        return True

    def add_element(x, elems, sum_bits, diff_bits):
        """Add x to set, return updated (elems, sum_bits, diff_bits)."""
        new_sum_bits = sum_bits | (1 << (x + x))
        new_diff_bits = diff_bits
        for a in elems:
            new_sum_bits |= (1 << (x + a))
            d = abs(x - a)
            new_diff_bits |= (1 << d)
        return elems + [x], new_sum_bits, new_diff_bits

    def rebuild(S):
        """Rebuild sum and diff bitmasks from a list."""
        sums = 0
        diffs = 0
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums |= (1 << (S[i] + S[j]))
            for j in range(len(S)):
                if i != j:
                    diffs |= (1 << abs(S[i] - S[j]))
        return sums, diffs

    def greedy_construct(order=None):
        """Greedy construction with optional custom order."""
        A = []
        B = []
        sumA = 0
        diffA = 0
        sumB = 0
        diffB = 0
        
        if order is None:
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
        else:
            candidates = order
        
        for x in candidates:
            if len(A) < MAX_EACH and is_valid_add(x, A, sumA, diffA, diffB):
                A, sumA, diffA = add_element(x, A, sumA, diffA)
                continue
            if len(B) < MAX_EACH and is_valid_add(x, B, sumB, diffB, diffA):
                B, sumB, diffB = add_element(x, B, sumB, diffB)
                continue
        
        return A, B

    def local_search(A, B):
        """1-for-2 swap local search."""
        if time.time() - start_time > DEADLINE - 8:
            return A, B
        
        sumA, diffA = rebuild(A)
        sumB, diffB = rebuild(B)
        used = set(A) | set(B)
        
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Try removing from A
            for i in range(len(A)):
                if time.time() - start_time > DEADLINE - 5:
                    break
                a = A[i]
                new_A = A[:i] + A[i+1:]
                new_sumA, new_diffA = rebuild(new_A)
                
                # Find all candidates valid for new_A
                valid_adds_A = []
                for x in range(1, N + 1):
                    if x in used and x != a:
                        continue
                    if len(new_A) < MAX_EACH and is_valid_add(x, new_A, new_sumA, new_diffA, diffB):
                        valid_adds_A.append(x)
                
                # Try 1-for-2: add two to A
                for j in range(len(valid_adds_A)):
                    x1 = valid_adds_A[j]
                    temp_A = new_A + [x1]
                    temp_sumA, temp_diffA = rebuild(temp_A)
                    for k in range(j + 1, len(valid_adds_A)):
                        x2 = valid_adds_A[k]
                        if len(temp_A) < MAX_EACH and is_valid_add(x2, temp_A, temp_sumA, temp_diffA, diffB):
                            A = temp_A + [x2]
                            sumA, diffA = rebuild(A)
                            used = set(A) | set(B)
                            improved = True
                            break
                    if improved:
                        break
                
                # Try 1-for-2: add one to A, one to B
                if not improved:
                    valid_adds_B = []
                    for x in range(1, N + 1):
                        if x in used and x != a:
                            continue
                        if len(B) < MAX_EACH and is_valid_add(x, B, sumB, diffB, new_diffA):
                            valid_adds_B.append(x)
                    
                    for x1 in valid_adds_A:
                        if len(new_A) < MAX_EACH:
                            temp_A = new_A + [x1]
                            temp_sumA, temp_diffA = rebuild(temp_A)
                            for x2 in valid_adds_B:
                                if is_valid_add(x2, B, sumB, diffB, temp_diffA):
                                    A = temp_A
                                    B = B + [x2]
                                    sumA, diffA = temp_sumA, temp_diffA
                                    sumB, diffB = rebuild(B)
                                    used = set(A) | set(B)
                                    improved = True
                                    break
                        if improved:
                            break
                
                if improved:
                    break
            
            # Try removing from B
            if not improved:
                for i in range(len(B)):
                    if time.time() - start_time > DEADLINE - 5:
                        break
                    b = B[i]
                    new_B = B[:i] + B[i+1:]
                    new_sumB, new_diffB = rebuild(new_B)
                    
                    valid_adds_B = []
                    for x in range(1, N + 1):
                        if x in used and x != b:
                            continue
                        if len(new_B) < MAX_EACH and is_valid_add(x, new_B, new_sumB, new_diffB, diffA):
                            valid_adds_B.append(x)
                    
                    # Try 1-for-2: add two to B
                    for j in range(len(valid_adds_B)):
                        x1 = valid_adds_B[j]
                        temp_B = new_B + [x1]
                        temp_sumB, temp_diffB = rebuild(temp_B)
                        for k in range(j + 1, len(valid_adds_B)):
                            x2 = valid_adds_B[k]
                            if len(temp_B) < MAX_EACH and is_valid_add(x2, temp_B, temp_sumB, temp_diffB, diffA):
                                B = temp_B + [x2]
                                sumB, diffB = rebuild(B)
                                used = set(A) | set(B)
                                improved = True
                                break
                        if improved:
                            break
                    
                    # Try 1-for-2: add one to B, one to A
                    if not improved:
                        valid_adds_A = []
                        for x in range(1, N + 1):
                            if x in used and x != b:
                                continue
                            if len(A) < MAX_EACH and is_valid_add(x, A, sumA, diffA, new_diffB):
                                valid_adds_A.append(x)
                        
                        for x1 in valid_adds_B:
                            if len(new_B) < MAX_EACH:
                                temp_B = new_B + [x1]
                                temp_sumB, temp_diffB = rebuild(temp_B)
                                for x2 in valid_adds_A:
                                    if is_valid_add(x2, A, sumA, diffA, temp_diffB):
                                        B = temp_B
                                        A = A + [x2]
                                        sumB, diffB = temp_sumB, temp_diffB
                                        sumA, diffA = rebuild(A)
                                        used = set(A) | set(B)
                                        improved = True
                                        break
                            if improved:
                                break
                    
                    if improved:
                        break
        
        return A, B

    # Main Loop: ILS with local search
    runs = 0
    while time.time() - start_time < DEADLINE - 5:
        A, B = greedy_construct()
        size = len(A) + len(B)
        
        # Apply local search if we have time and this is competitive
        if time.time() - start_time < DEADLINE - 12 and size >= best_size:
            A, B = local_search(A, B)
            size = len(A) + len(B)
        
        if size > best_size:
            best_size = size
            best_A = A
            best_B = B
        
        runs += 1
        if best_size >= 128:
            break

    # Ensure non-empty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    return [best_A, best_B]

if __name__ == "__main__":
    # Quick test
    A, B = construct()
    print(f"Size A: {len(A)}, Size B: {len(B)}, Total: {len(A)+len(B)}")
    # Verify
    def verify(A, B):
        # Check Sidon
        def is_sidon(S):
            sums = set()
            for i in range(len(S)):
                for j in range(i, len(S)):
                    s = S[i] + S[j]
                    if s in sums:
                        return False
                    sums.add(s)
            return True
        
        if not is_sidon(A) or not is_sidon(B):
            return False, "Sidon check failed"
            
        # Check Disjoint Diffs
        def get_diffs(S):
            return {abs(a-b) for i, a in enumerate(S) for b in S[i+1:]}
        
        dA = get_diffs(A)
        dB = get_diffs(B)
        
        if dA & dB:
            return False, f"Diff intersection: {dA & dB}"
            
        return True, "Valid"

    valid, msg = verify(A, B)
    print(f"Valid: {valid}, Msg: {msg}")
