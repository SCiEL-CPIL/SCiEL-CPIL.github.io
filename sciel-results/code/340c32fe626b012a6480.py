import random
import time

def construct():
    n = 256
    max_size = 64 # actually limited by Sidon property to ~16
    deadline = time.time() + 55.0
    
    best_sum = 0
    best_A = []
    best_B = []
    
    # Helper to check Sidon property
    def is_sidon(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
        
    def get_diffs(S):
        diffs = set()
        for i in range(len(S)):
            for j in range(len(S)):
                if i != j:
                    diffs.add(S[i] - S[j])
        return diffs

    # We aim for |A|=16, |B|=16.
    # Max possible sum is 32.
    
    for restart in range(500):
        if time.time() > deadline:
            break
            
        # Try to build A and B
        # Randomize the order of candidates or the sets themselves?
        # Greedy construction of Sidon set is sensitive to order.
        
        # Let's try to construct A first, then B.
        # Random subset of [1, 256]
        candidates = list(range(1, n+1))
        random.shuffle(candidates)
        
        A = []
        B = []
        
        # Greedy for A
        for x in candidates:
            if len(A) >= 16: break
            # Check if adding x keeps A Sidon
            # And check difference conflict with current B? 
            # Since B is empty, just Sidon check.
            # But we also need to reserve differences for B?
            # Hard to know in advance.
            
            # Simple check: is A + [x] Sidon?
            new_A = A + [x]
            if is_sidon(new_A):
                A = new_A
        
        # Now construct B
        # We need B to be Sidon AND D_B disjoint from D_A.
        D_A = get_diffs(A)
        
        for x in candidates:
            if len(B) >= 16: break
            new_B = B + [x]
            if is_sidon(new_B):
                D_new_B = get_diffs(new_B)
                # Check disjoint
                if D_new_B.isdisjoint(D_A):
                    B = new_B
                    # Update D_A? No, D_A is fixed.
                    # But we need to ensure future additions to B don't conflict with D_A.
                    # The check `D_new_B.isdisjoint(D_A)` ensures the *current* B is ok.
                    # But does it ensure we can extend? No.
                    # However, if we greedily pick, we might paint ourselves into a corner.
                    
        current_sum = len(A) + len(B)
        if current_sum > best_sum:
            best_sum = current_sum
            best_A = A[:]
            best_B = B[:]
            
        if best_sum == 32:
            break
            
    return [best_A, best_B]
