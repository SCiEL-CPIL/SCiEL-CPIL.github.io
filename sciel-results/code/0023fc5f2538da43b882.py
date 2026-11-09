import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50 seconds safety margin
    
    best_seq = []
    best_len = 0
    
    # We want to find the longest sequence.
    # Candidates are 2..128.
    # We use backtracking.
    
    def check_conflicts(seq, new_val, used_prods):
        """
        Checks if adding new_val to seq creates any duplicate product.
        Returns True if conflict exists, False otherwise.
        Also returns the list of new products if no conflict, to update used_prods.
        """
        # Suffixes of seq
        # We need products of (suffix + new_val)
        # Let seq = [a1, ..., ak]
        # Suffixes: [ak], [ak-1, ak], ..., [a1, ..., ak]
        # New products: new_val * ak, new_val * ak-1 * ak, ..., new_val * prod(seq)
        
        prod = new_val
        new_prods = []
        # Iterate from end of seq to start
        for i in range(len(seq) - 1, -1, -1):
            new_prods.append(prod)
            prod *= seq[i]
            # Optimization: if prod exceeds some limit? 
            # Max product is 128^128 which is huge, but Python handles big ints.
            # However, if prod is already in used_prods, we can stop? 
            # No, we need to check all. But if one conflicts, we return True.
            if prod in used_prods:
                return True, None
        # Also check if new_val itself is in used_prods
        # This is covered by the first iteration if seq is not empty?
        # If seq is empty, loop doesn't run. We must check new_val separately.
        if new_val in used_prods:
            return True, None
            
        # If no conflict found in loop, and new_val not in used_prods
        # Wait, the loop adds prod to new_prods BEFORE checking?
        # Let's re-structure.
        
        # Let's just compute all and check.
        # Actually, the above loop logic was slightly off.
        # Let's do it cleanly.
        
        pass # Placeholder, will rewrite in main function

    def search(seq, used_prods, start_idx):
        nonlocal best_seq, best_len, deadline
        
        # Time check
        if time.time() > deadline:
            return

        # Update best
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            # Optional: print progress to stderr if needed, but keep it quiet.
            # print(f"Found length {best_len}", file=sys.stderr)

        # Pruning: Max possible length
        # We can pick at most (128 - seq[-1]) more numbers if we pick consecutive?
        # No, we can pick any numbers > seq[-1].
        # Max remaining candidates = 128 - seq[-1] (if seq not empty) else 127
        if not seq:
            max_remaining = 127
        else:
            max_remaining = 128 - seq[-1]
            
        if len(seq) + max_remaining <= best_len:
            return

        # Try adding candidates
        # Heuristic: Try smaller numbers first to maximize length
        # Range: from start_idx to 128
        for x in range(start_idx, 129):
            # Check conflicts
            # New products are x * (product of suffix of seq)
            # Suffixes of seq: 
            # If seq is [a, b, c], suffixes are [c], [b,c], [a,b,c]
            # Products: c, b*c, a*b*c
            # New products: x*c, x*b*c, x*a*b*c
            # Also x itself (suffix of length 0? No, singleton block [x])
            
            # Let's compute suffix products of seq
            # We can maintain a list of suffix products to speed this up?
            # Or just compute on the fly.
            
            conflict = False
            new_prods = set()
            
            # Singleton
            if x in used_prods:
                conflict = True
            else:
                new_prods.add(x)
                
            # Longer suffixes
            if not conflict and seq:
                prod = x
                # Iterate backwards through seq
                for i in range(len(seq) - 1, -1, -1):
                    prod *= seq[i]
                    if prod in used_prods:
                        conflict = True
                        break
                    new_prods.add(prod)
            
            if conflict:
                continue
            
            # Valid move
            # Recurse
            # We need to update used_prods. 
            # To avoid copying the large set, we can add, recurse, remove.
            used_prods.update(new_prods)
            seq.append(x)
            
            search(seq, used_prods, x + 1)
            
            seq.pop()
            used_prods.difference_update(new_prods)
            
            # Time check after recursion
            if time.time() > deadline:
                return

    # Initial call
    # We can start with any number. 
    # Since we want max length, starting with smaller numbers is generally better.
    # But the search handles this by trying all start_idx.
    # However, if we just call search([], set(), 2), it will try 2, then 3, etc.
    # This is a standard DFS.
    
    search([], set(), 2)
    
    return best_seq

# EVOLVE-BLOCK-END
