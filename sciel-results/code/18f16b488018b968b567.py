import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110
    
    total_words = q ** n
    
    # Precompute balls
    balls = []
    for c in range(total_words):
        digits = []
        temp = c
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        digits.reverse()
        
        covered = [c]
        for i in range(n):
            original = digits[i]
            for val in range(q):
                if val == original:
                    continue
                new_digits = digits[:]
                new_digits[i] = val
                w = 0
                for d in new_digits:
                    w = w * q + d
                covered.append(w)
        balls.append(covered)
        
    # Helper to check if a set of codewords is a valid covering
    # We maintain cover_count globally, so we just need to check min(cover_count) >= 1
    # But during local search, we modify cover_count.
    
    # Greedy Construction
    cover_count = [0] * total_words
    uncovered = [True] * total_words
    uncovered_count = total_words
    code = []
    
    while uncovered_count > 0:
        if time.time() > deadline:
            break
        best_c = -1
        max_new_covers = 0
        for c in range(total_words):
            cnt = 0
            for w in balls[c]:
                if uncovered[w]:
                    cnt += 1
            if cnt > max_new_covers:
                max_new_covers = cnt
                best_c = c
                
        if best_c == -1:
            break
        code.append(best_c)
        for w in balls[best_c]:
            if uncovered[w]:
                uncovered[w] = False
                uncovered_count -= 1
            cover_count[w] += 1
            
    # Removal Phase
    def try_remove_redundant(code, cover_count):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                redundant = True
                for w in balls[c]:
                    if cover_count[w] <= 1:
                        redundant = False
                        break
                if redundant:
                    code.pop(i)
                    for w in balls[c]:
                        cover_count[w] -= 1
                    changed = True
                    break
        return code, cover_count

    code, cover_count = try_remove_redundant(code, cover_count)
    
    # Local Search
    # We try to swap a codeword with another to potentially unlock removals.
    # We only accept a swap if it leads to a valid covering AND allows a removal.
    # Or, we can accept a swap if it keeps the count same but "improves" some metric?
    # Simplest: Only accept if it leads to a lower count after removal.
    
    best_code_len = len(code)
    
    # We need a way to quickly check if a swap is valid.
    # Swap c_old with c_new.
    # Decrement counts for c_old, increment for c_new.
    # Check if any count becomes 0.
    # If yes, invalid.
    
    while time.time() < deadline:
        improved = False
        # Try a few random swaps
        for _ in range(100): # Limit swaps per outer loop iteration
            if time.time() > deadline:
                break
            if len(code) < 2:
                break
            i = random.randint(0, len(code) - 1)
            c_old = code[i]
            c_new = random.randint(0, total_words - 1)
            if c_new == c_old:
                continue
            # Check if c_new is already in code?
            # It doesn't matter for validity, but duplicates are useless.
            # We can allow it, removal will handle it.
            
            # Tentative Swap
            for w in balls[c_old]:
                cover_count[w] -= 1
            for w in balls[c_new]:
                cover_count[w] += 1
                
            # Check validity
            valid = True
            for w in balls[c_old]:
                if cover_count[w] == 0:
                    valid = False
                    break
            # Also need to check words covered by c_new that might have been 0 before?
            # No, if c_old was the only cover for some w, count becomes 0.
            # If c_new covers some w, count increases.
            # We only need to check words where count might have dropped to 0.
            # Those are in balls[c_old].
            
            if not valid:
                # Revert
                for w in balls[c_old]:
                    cover_count[w] += 1
                for w in balls[c_new]:
                    cover_count[w] -= 1
                continue
            
            # If valid, try to remove redundants
            new_code = code[:]
            new_code[i] = c_new
            new_cover_count = cover_count[:]
            
            new_code, new_cover_count = try_remove_redundant(new_code, new_cover_count)
            
            if len(new_code) < best_code_len:
                # Accept
                code = new_code
                cover_count = new_cover_count
                best_code_len = len(code)
                improved = True
                break # Restart outer loop with new best
            else:
                # Revert
                for w in balls[c_old]:
                    cover_count[w] += 1
                for w in balls[c_new]:
                    cover_count[w] -= 1
                # Note: try_remove_redundant modified new_cover_count, 
                # but we reverted cover_count manually. 
                # We must ensure cover_count is back to state before swap.
                # The revert logic above restores the pre-swap state.
                
        if not improved:
            # If no improvement in this batch, maybe break or continue?
            # Let's continue for a bit, but if we are stuck for a long time, break.
            # For simplicity, we just loop until deadline.
            pass

    # Convert code back to list of lists
    result = []
    for c in code:
        digits = []
        temp = c
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        digits.reverse()
        result.append(digits)
        
    return result
