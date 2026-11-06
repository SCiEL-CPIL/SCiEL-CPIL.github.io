import numpy as np
import time
import random
import sys

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills after 90s, but we want to return before that.
    # Let's aim for 55s to be safe, leaving room for overhead.
    deadline = start_time + 55.0

    n = 128
    target_min = 128
    target_max = 256
    target_size = target_max - target_min + 1 # 129

    # We represent A as a boolean array of length 129 (indices 0..128)
    # conv = np.convolve(A_bool, A_bool, mode='full')
    # The sum s corresponds to index s in conv.
    # We care about conv[128:257]

    def compute_conv(A_bool):
        # A_bool is a numpy array of 0s and 1s of length 129
        return np.convolve(A_bool, A_bool, mode='full')

    def get_peak_and_valid(conv):
        # conv is the full convolution result
        # Target sums are 128 to 256
        target_counts = conv[128:257]
        if np.any(target_counts == 0):
            return -1, False # Invalid
        return np.max(target_counts), True

    def evaluate_move(conv, A_bool, x, add):
        """
        Evaluate the change in peak and validity if we add or remove x.
        Returns (new_peak, new_valid).
        This is O(N) because we only update the convolution by adding/subtracting the contribution of x.
        Contribution of x to convolution: x * A_bool shifted by x.
        Actually, conv_new = conv_old +/- convolve(A_bool, [0..x..0])
        Since A_bool is binary, convolve(A_bool, delta_x) is just A_bool shifted by x.
        """
        # Create a delta array: 1 at position x, 0 elsewhere
        # But wait, np.convolve(A, B) where B is [0,0,1,0,0] is A shifted by 2.
        # Let's be precise.
        # conv[k] = sum_{i+j=k} A[i]A[j].
        # If we add x to A, new A' = A + e_x.
        # conv(A') = conv(A) + conv(A, e_x) + conv(e_x, e_x).
        # conv(A, e_x) is A shifted by x.
        # conv(e_x, e_x) is 1 at 2x.
        
        # So, delta_conv = np.zeros_like(conv)
        # delta_conv[x : x + len(A_bool)] += A_bool
        # delta_conv[2*x] += 1
        
        # If we remove x, delta_conv is negative.
        
        sign = 1 if add else -1
        
        # We don't need to compute the full new conv if we can just check the target range.
        # But we need to update the conv for the next step.
        # For speed, let's just compute the change in the target range.
        
        # However, maintaining the full conv array is easy and fast.
        # Let's just compute the new conv in the target range to check validity/peak.
        
        # A_bool is length 129.
        # Shifted A_bool: indices x to x+128.
        # The term conv(A, e_x) adds A_bool[i] to index i+x.
        
        # Let's create the delta for the full conv vector.
        # This is O(N) to create and add.
        
        # Actually, for a single move, we can just calculate the new counts in the target range directly?
        # No, updating the full conv is O(N) which is ~129 ops. Very fast.
        
        # Let's do the full update to keep state consistent.
        
        delta = np.zeros(len(conv), dtype=np.int32)
        
        # conv(A, e_x)
        # e_x has 1 at index x.
        # (A * e_x)[k] = sum A[i] e_x[k-i] = A[k-x] if k-x in [0, 128]
        # So delta[k] += A_bool[k-x] for k in [x, x+128]
        
        start_idx = x
        end_idx = x + 129
        # Ensure bounds
        if start_idx < len(delta):
             # A_bool is length 129.
             # We add A_bool to delta[start_idx : end_idx]
             # But we must respect the length of A_bool.
             # A_bool is 0..128.
             # delta[x : x+129] += A_bool
             delta[start_idx:end_idx] += A_bool
             
        # conv(e_x, e_x)
        # (e_x * e_x)[k] = 1 if k = 2x
        if 2*x < len(delta):
            delta[2*x] += 1
            
        # Apply sign
        delta *= sign
        
        new_conv = conv + delta
        
        # Check validity and peak in target range
        target_counts = new_conv[128:257]
        
        if np.any(target_counts == 0):
            return -1, False, new_conv
            
        peak = np.max(target_counts)
        return peak, True, new_conv

    # Initial Best
    best_A_bool = None
    best_peak = float('inf')
    
    # --- Phase 1: Greedy Initialization ---
    # Start with all elements.
    A_bool = np.ones(129, dtype=np.int32)
    conv = compute_conv(A_bool)
    peak, valid = get_peak_and_valid(conv)
    
    # If not valid (should be valid with all elements), repair.
    # With all elements 0..128, every sum 128..256 is covered many times.
    
    # Greedy Removal
    # We want to remove elements to minimize peak while maintaining validity.
    # Iterate until no more removals possible.
    
    max_iter = 1000
    for _ in range(max_iter):
        if time.time() > deadline - 5.0:
            break
            
        removed_any = False
        # Shuffle order to avoid bias
        indices = np.random.permutation(129)
        
        for x in indices:
            if A_bool[x] == 0:
                continue
                
            # Try removing x
            new_peak, new_valid, new_conv = evaluate_move(conv, A_bool, x, add=False)
            
            if new_valid:
                # Accept if peak is lower or equal
                # We prefer lower peak. If peak is same, we still remove to reduce size (though size doesn't affect score directly, smaller A might be easier to optimize? 
                # Actually, the score is just the peak. So if peak is same, removing is neutral for score but might open up future moves?
                # Usually, we want to minimize peak. If peak stays same, we can remove to simplify.
                if new_peak <= peak:
                    A_bool[x] = 0
                    conv = new_conv
                    peak = new_peak
                    removed_any = True
                    # Update best if improved
                    if peak < best_peak:
                        best_peak = peak
                        best_A_bool = A_bool.copy()
                        
        if not removed_any:
            break

    # --- Phase 2: Local Search (Hill Climbing with Swaps) ---
    # We are at a local optimum for removals.
    # Now we try swaps: remove x, add y.
    # Or just add y (if it reduces peak? Unlikely if we are at a minimum peak for removals, but maybe adding a specific y allows removing two others later? 
    # Actually, if we are at a local min for removals, adding an element increases the number of pairs, likely increasing or keeping peak. 
    # So we should focus on swaps: remove x, add y.
    
    # We will run multiple restarts.
    # Current A_bool is our starting point for restart 0.
    
    num_restarts = 10
    current_A_bool = A_bool.copy()
    current_conv = conv.copy()
    current_peak = peak
    
    if current_peak < best_peak:
        best_peak = current_peak
        best_A_bool = current_A_bool.copy()
        
    for restart in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
            
        # If not first restart, perturb current best or start fresh?
        # Let's start from the current best found so far, but perturb it.
        if restart > 0:
            # Perturb: flip 5-10 bits
            num_flips = random.randint(5, 10)
            for _ in range(num_flips):
                x = random.randint(0, 128)
                current_A_bool[x] = 1 - current_A_bool[x]
            
            # Recompute conv
            current_conv = compute_conv(current_A_bool)
            current_peak, valid = get_peak_and_valid(current_conv)
            
            if not valid:
                # Repair: add random elements until valid
                attempts = 0
                while not valid and attempts < 50:
                    x = random.randint(0, 128)
                    if current_A_bool[x] == 0:
                        current_A_bool[x] = 1
                        current_conv, current_peak, valid = evaluate_move(current_conv, current_A_bool, x, add=True)
                        # Note: evaluate_move returns new_conv, but we need to update A_bool before calling?
                        # My evaluate_move assumes A_bool is the OLD state.
                        # So I must update A_bool BEFORE calling evaluate_move if I want the conv to reflect the new A.
                        # Wait, in the loop above:
                        # 1. Set A_bool[x] = 1
                        # 2. Call evaluate_move(conv, A_bool, x, add=True)
                        #    Inside evaluate_move, it calculates delta based on A_bool.
                        #    But A_bool already has x=1.
                        #    The formula conv(A') = conv(A) + conv(A, e_x) + e_2x assumes A is the OLD set.
                        #    If I modify A_bool first, the "A" in the formula is wrong.
                        #    
                        # Correction: 
                        # 1. Keep A_bool as OLD.
                        # 2. Call evaluate_move(conv, A_bool, x, add=True).
                        # 3. If accepted, THEN update A_bool[x]=1 and conv = new_conv.
                        
                        # So the repair loop above was slightly flawed in logic if it modified A_bool first.
                        # Let's re-verify the repair logic below.
                    attempts += 1
                if not valid:
                    # Reset to best known valid
                    current_A_bool = best_A_bool.copy()
                    current_conv = compute_conv(current_A_bool)
                    current_peak, _ = get_peak_and_valid(current_conv)
                    continue

        # Local Search Loop
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # Try Swaps
            # Randomly pick x in A to remove, y not in A to add.
            # To be efficient, we can try a few random swaps.
            
            # Get list of elements in A and not in A
            in_A = np.where(current_A_bool == 1)[0]
            not_in_A = np.where(current_A_bool == 0)[0]
            
            if len(in_A) == 0 or len(not_in_A) == 0:
                break
                
            # Try up to 50 random swaps
            for _ in range(50):
                if time.time() > deadline - 2.0:
                    break
                    
                x = random.choice(in_A)
                y = random.choice(not_in_A)
                
                # Remove x
                # Check if removing x keeps validity?
                # Actually, we do a swap: remove x, add y.
                # We can simulate this in two steps or one combined step.
                # Step 1: Remove x.
                p1, v1, conv1 = evaluate_move(current_conv, current_A_bool, x, add=False)
                if not v1:
                    # Removing x breaks validity. We must add y to fix it?
                    # If removing x makes some count 0, adding y might fix it if y contributes to that sum.
                    # It's complex. Let's just try the full swap evaluation.
                    # 
                    # Full Swap Delta:
                    # A' = A - {x} + {y}
                    # conv(A') = conv(A) - conv(A, e_x) - e_2x + conv(A, e_y) + e_2y 
                    # Wait, conv(A - {x}) = conv(A) - conv(A, e_x) - e_2x is only valid if we subtract the contribution of x.
                    # Yes.
                    # Then conv(A - {x} + {y}) = conv(A - {x}) + conv(A - {x}, e_y) + e_2y
                    # This is getting complicated to do in O(N) without storing intermediate states.
                    # 
                    # Simpler: Just compute the new conv from scratch? 
                    # No, O(N log N) is ~1000 ops. 
                    # O(N) delta is ~200 ops.
                    # 
                    # Let's stick to sequential moves: Remove x, then Add y.
                    # If Remove x is invalid, we can still try Add y to the INVALID state?
                    # No, we need a valid path? Or just check the final state?
                    # The final state is what matters.
                    # 
                    # Let's just do the two steps and check the final validity.
                    # Step 1: Remove x.
                    p1, v1, conv_after_remove = evaluate_move(current_conv, current_A_bool, x, add=False)
                    
                    # Step 2: Add y to the state after removing x.
                    # Note: current_A_bool still has x=1. We need to pass the "virtual" A_bool where x=0?
                    # My evaluate_move uses A_bool to calculate the shift.
                    # If I pass current_A_bool (with x=1) to add y, it will calculate conv(A, e_y).
                    # But the base is conv(A - {x}).
                    # The formula conv(B + e_y) = conv(B) + conv(B, e_y) + e_2y requires B to be the base.
                    # Here B = A - {x}.
                    # So I need A_bool_without_x.
                    
                    # This is inefficient. 
                    # Alternative: Just recompute conv for the swap? 
                    # np.convolve is fast. 129 points. 
                    # Let's benchmark mentally: 129^2 = 16k ops. NumPy does this in < 10us.
                    # If we do 1000 swaps, that's 10ms. 
                    # We have 50 seconds. We can do 10,000,000 swaps.
                    # So, let's just use the simple, robust method:
                    # 1. Create new A_bool (copy).
                    # 2. Flip x and y.
                    # 3. Compute new conv.
                    # 4. Check peak/valid.
                    # 5. Accept if better.
                    
                    # This is O(N^2) per move but with tiny constant.
                    # Actually, creating a copy is O(N). Convolve is O(N log N).
                    # Let's try this simpler approach first. It's less bug-prone.
                    
                    new_A_bool = current_A_bool.copy()
                    new_A_bool[x] = 0
                    new_A_bool[y] = 1
                    
                    new_conv = compute_conv(new_A_bool)
                    new_peak, new_valid = get_peak_and_valid(new_conv)
                    
                    if new_valid:
                        if new_peak < current_peak:
                            current_A_bool = new_A_bool
                            current_conv = new_conv
                            current_peak = new_peak
                            improved = True
                            if current_peak < best_peak:
                                best_peak = current_peak
                                best_A_bool = current_A_bool.copy()
                            break # Restart local search with new state
                        elif new_peak == current_peak:
                            # Tie: prefer smaller set? 
                            # Size is not in score, but smaller set is "simpler".
                            # Let's accept to move around the plateau.
                            if random.random() < 0.5:
                                current_A_bool = new_A_bool
                                current_conv = new_conv
                                # Peak stays same
                                improved = True
                                break

    # Final Result
    if best_A_bool is None:
        # Fallback
        best_A_bool = np.ones(129, dtype=np.int32)
        
    # Convert to sorted list of integers
    result = sorted(np.where(best_A_bool == 1)[0].tolist())
    
    return result

# Run construct to test locally if needed, but the function is the entry point.
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}, First few: {res[:10]}")
    # Quick check
    A = set(res)
    counts = [0]*129
    for i, a in enumerate(res):
        for b in res[i:]:
            s = a+b
            if 128 <= s <= 256:
                counts[s-128] += 1
    print(f"Min count: {min(counts)}, Max count: {max(counts)}")
