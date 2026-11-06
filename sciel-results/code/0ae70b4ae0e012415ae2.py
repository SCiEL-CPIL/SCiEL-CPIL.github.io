"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 55.0  # Safety margin
    
    N = 512
    
    # Precompute: for each element x (1..N), find all 4-APs containing x
    # conflict[x] = list of tuples (a,b,c,d) where x is one of the terms
    # We'll use a more efficient approach: for each x, precompute which other elements
    # cannot be simultaneously present with x in a way that forms a 4-AP.
    
    # Actually, let's use a bitset approach for speed.
    # Represent the set as a numpy bit array of length 512 (indices 0..511 for elements 1..512)
    
    import numpy as np
    
    def _can_add(bits, x):
        """
        Check if element x (1-indexed) can be added to the set represented by bits
        without creating a 4-AP.
        """
        idx = x - 1  # 0-indexed
        
        # Case 1: x is the 4th term: x-3d, x-2d, x-d, x
        # Need all three of x-3d, x-2d, x-d to be in the set
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if bits[x-d-1] and bits[x-2*d-1] and bits[x-3*d-1]:
                return False
        
        # Case 2: x is the 3rd term: x-2d, x-d, x, x+d
        max_d = (x - 1) // 2
        for d in range(1, max_d + 1):
            if x + d <= N and bits[x+d-1] and bits[x-d-1] and bits[x-2*d-1]:
                return False
        
        # Case 3: x is the 2nd term: x-d, x, x+d, x+2d
        max_d = min(x - 1, (N - x) // 2)
        for d in range(1, max_d + 1):
            if bits[x+d-1] and bits[x+2*d-1] and bits[x-d-1]:
                return False
        
        # Case 4: x is the 1st term: x, x+d, x+2d, x+3d
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if bits[x+d-1] and bits[x+2*d-1] and bits[x+3*d-1]:
                return False
        
        return True
    
    def _is_4ap_free(bits):
        """Check if the set represented by bits is 4-AP free."""
        for x in range(1, N + 1):
            if not bits[x-1]:
                continue
            # Temporarily remove x and check if any 4-AP containing x exists
            bits[x-1] = 0
            conflict = False
            idx = x - 1
            
            # Case 1: x is the 4th term
            max_d = (x - 1) // 3
            for d in range(1, max_d + 1):
                if bits[x-d-1] and bits[x-2*d-1] and bits[x-3*d-1]:
                    conflict = True
                    break
            if conflict:
                bits[x-1] = 1
                return False
            
            # Case 2: x is the 3rd term
            max_d = (x - 1) // 2
            for d in range(1, max_d + 1):
                if x + d <= N and bits[x+d-1] and bits[x-d-1] and bits[x-2*d-1]:
                    conflict = True
                    break
            if conflict:
                bits[x-1] = 1
                return False
            
            # Case 3: x is the 2nd term
            max_d = min(x - 1, (N - x) // 2)
            for d in range(1, max_d + 1):
                if bits[x+d-1] and bits[x+2*d-1] and bits[x-d-1]:
                    conflict = True
                    break
            if conflict:
                bits[x-1] = 1
                return False
            
            # Case 4: x is the 1st term
            max_d = (N - x) // 3
            for d in range(1, max_d + 1):
                if bits[x+d-1] and bits[x+2*d-1] and bits[x+3*d-1]:
                    conflict = True
                    break
            if conflict:
                bits[x-1] = 1
                return False
            
            bits[x-1] = 1
        
        return True
    
    def _count_ap_conflicts(bits, x):
        """Count how many 4-APs would be formed if x were added to the set."""
        count = 0
        idx = x - 1
        
        # Case 1: x is the 4th term
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if bits[x-d-1] and bits[x-2*d-1] and bits[x-3*d-1]:
                count += 1
        
        # Case 2: x is the 3rd term
        max_d = (x - 1) // 2
        for d in range(1, max_d + 1):
            if x + d <= N and bits[x+d-1] and bits[x-d-1] and bits[x-2*d-1]:
                count += 1
        
        # Case 3: x is the 2nd term
        max_d = min(x - 1, (N - x) // 2)
        for d in range(1, max_d + 1):
            if bits[x+d-1] and bits[x+2*d-1] and bits[x-d-1]:
                count += 1
        
        # Case 4: x is the 1st term
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if bits[x+d-1] and bits[x+2*d-1] and bits[x+3*d-1]:
                count += 1
        
        return count
    
    # Greedy initialization
    bits = np.zeros(N, dtype=np.uint8)
    for x in range(1, N + 1):
        if _can_add(bits, x):
            bits[x-1] = 1
    
    best_bits = bits.copy()
    best_size = int(bits.sum())
    
    # Local search with swap moves
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add any element
        candidates = [x for x in range(1, N + 1) if not bits[x-1]]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            if _can_add(bits, x):
                bits[x-1] = 1
                best_size = int(bits.sum())
                best_bits = bits.copy()
                improved = True
                break
        
        # If no simple add worked, try swap moves: remove one element, try to add another
        if not improved and time.time() < deadline:
            # Try removing one element and adding a different one
            selected = [x for x in range(1, N + 1) if bits[x-1]]
            random.shuffle(selected)
            
            for y in selected:
                if time.time() > deadline:
                    break
                
                # Temporarily remove y
                bits[y-1] = 0
                
                # Try to add any element that was previously blocked
                candidates = [x for x in range(1, N + 1) if not bits[x-1] and x != y]
                random.shuffle(candidates)
                
                added = False
                for x in candidates:
                    if time.time() > deadline:
                        break
                    
                    if _can_add(bits, x):
                        bits[x-1] = 1
                        if int(bits.sum()) > best_size:
                            best_size = int(bits.sum())
                            best_bits = bits.copy()
                            improved = True
                        added = True
                        break
                
                # Restore y if we didn't improve
                if not improved or not added:
                    bits[y-1] = 1
                elif not added:
                    # We removed y but didn't add anything, restore
                    bits[y-1] = 1
        
        # Try remove-one-add-two moves
        if not improved and time.time() < deadline:
            selected = [x for x in range(1, N + 1) if bits[x-1]]
            random.shuffle(selected)
            
            for y in selected:
                if time.time() > deadline:
                    break
                
                # Temporarily remove y
                bits[y-1] = 0
                saved_bits = bits.copy()
                
                # Try to add as many elements as possible
                added_count = 0
                candidates = [x for x in range(1, N + 1) if not bits[x-1] and x != y]
                random.shuffle(candidates)
                
                for x in candidates:
                    if time.time() > deadline:
                        break
                    if _can_add(bits, x):
                        bits[x-1] = 1
                        added_count += 1
                        if added_count >= 2:
                            if int(bits.sum()) > best_size:
                                best_size = int(bits.sum())
                                best_bits = bits.copy()
                                improved = True
                            break
                
                # If we didn't improve, restore
                if not improved:
                    bits = saved_bits.copy()
    
    # Final verification and cleanup
    bits = best_bits
    
    # Convert to list
    result = [x for x in range(1, N + 1) if bits[x-1]]
    
    # Verify
    if not _is_4ap_free(bits.copy()):
        # Fallback to a known valid set
        return [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    return sorted(result)
# EVOLVE-BLOCK-END
