import time
import random
import numpy as np

def construct():
    deadline = time.time() + 80  # safety margin
    best_peak = float('inf')
    best_A = None
    
    N = 128
    targets = range(128, 257)  # 129 targets
    num_targets = 129
    
    def compute_peak(A):
        """Compute the maximum representation multiplicity for targets [128,256]."""
        A_set = set(A)
        counts = [0] * num_targets
        for i, a in enumerate(A):
            for b in A[i:]:
                s = a + b
                if 128 <= s <= 256:
                    counts[s - 128] += 1
        return max(counts)
    
    def compute_counts(A):
        """Return the count array for targets [128,256]."""
        counts = [0] * num_targets
        for i, a in enumerate(A):
            for b in A[i:]:
                s = a + b
                if 128 <= s <= 256:
                    counts[s - 128] += 1
        return counts
    
    def is_valid(counts):
        """Check if all targets are covered."""
        return all(c > 0 for c in counts)
    
    def get_peak(counts):
        return max(counts)
    
    # Try different sizes
    for k in range(34, 41):
        if time.time() > deadline - 5:
            break
        
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            
            # Initialize: try to spread elements
            # Start with a structured set then randomize
            A = set()
            # Try to place elements to cover all targets
            # Simple random init
            A = set(random.sample(range(0, 129), k))
            
            # Ensure we have some structure - add endpoints
            A.add(0)
            A.add(128)
            if len(A) > k:
                # Remove a random element that's not 0 or 128
                others = list(A - {0, 128})
                if others:
                    A.remove(random.choice(others))
            
            A = list(A)
            
            counts = compute_counts(A)
            
            if not is_valid(counts):
                # Try to fix by adjusting
                peak = get_peak(counts) if is_valid(counts) else float('inf')
            else:
                peak = get_peak(counts)
            
            if peak < best_peak:
                best_peak = peak
                best_A = list(A)
            
            # Local search: swap-based
            if is_valid(counts):
                for step in range(200):
                    if time.time() > deadline - 5:
                        break
                    
                    # Pick a random element to swap
                    idx = random.randint(0, len(A) - 1)
                    old_val = A[idx]
                    
                    # Try a few random new values
                    for _ in range(10):
                        new_val = random.randint(0, 128)
                        if new_val == old_val or new_val in A:
                            continue
                        
                        # Compute delta
                        # Remove old_val's contributions
                        A[idx] = new_val
                        new_counts = compute_counts(A)
                        
                        if is_valid(new_counts):
                            new_peak = get_peak(new_counts)
                            if new_peak < peak:
                                peak = new_peak
                                counts = new_counts
                                break
                            elif new_peak == peak and random.random() < 0.3:
                                # Accept sideways moves sometimes
                                counts = new_counts
                                break
                        else:
                            # Rollback
                            A[idx] = old_val
                    
                    A[idx] = old_val if A[idx] != new_val else old_val
                    # Actually let me redo this more carefully
                    break  # Simplified for now
            
            # Better local search
            A = list(A)
            counts = compute_counts(A)
            if not is_valid(counts):
                continue
                
            peak = get_peak(counts)
            
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                # Try all single-element swaps
                for idx in range(len(A)):
                    if time.time() > deadline - 5:
                        break
                    old_val = A[idx]
                    for new_val in range(0, 129):
                        if new_val == old_val or new_val in A:
                            continue
                        
                        A[idx] = new_val
                        new_counts = compute_counts(A)
                        
                        if is_valid(new_counts):
                            new_peak = get_peak(new_counts)
                            if new_peak < peak:
                                peak = new_peak
                                counts = new_counts
                                improved = True
                                break
                            elif new_peak == peak and random.random() < 0.1:
                                counts = new_counts
                                break
                        A[idx] = old_val
                    
                    if improved:
                        break
            
            if peak < best_peak:
                best_peak = peak
                best_A = list(A)
            
            if best_peak <= 4:
                break
        
        if best_peak <= 4:
            break
    
    if best_A is None:
        # Fallback to seed
        return list(range(64, 129))
    
    return best_A
