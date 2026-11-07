import time
import random

def construct():
    deadline = time.time() + 80
    
    n = 128
    max_sum = 2 * n
    
    def is_valid(S):
        """Check if S has no forbidden triple."""
        mem = [False] * (max_sum + 1)
        for x in S:
            mem[x] = True
        elems = sorted(S)
        m = len(elems)
        for i in range(m):
            a = elems[i]
            for j in range(i + 1, m):
                b = elems[j]
                ab = a + b
                if ab > max_sum or not mem[ab]:
                    continue
                for k in range(j + 1, m):
                    c = elems[k]
                    ac = a + c
                    if ac > max_sum or not mem[ac]:
                        continue
                    bc = b + c
                    if bc <= max_sum and mem[bc]:
                        return False
        return True
    
    def find_bad_triples(S, mem):
        bad = []
        elems = sorted(S)
        m = len(elems)
        for i in range(m):
            a = elems[i]
            for j in range(i + 1, m):
                b = elems[j]
                ab = a + b
                if ab > max_sum or not mem[ab]:
                    continue
                for k in range(j + 1, m):
                    c = elems[k]
                    ac = a + c
                    if ac > max_sum or not mem[ac]:
                        continue
                    bc = b + c
                    if bc <= max_sum and mem[bc]:
                        bad.append((a, b, c))
        return bad
    
    def count_involvement(bad_triples, S):
        counts = {}
        for (a, b, c) in bad_triples:
            for x in (a, b, c, a + b, a + c, b + c):
                if x in S:
                    counts[x] = counts.get(x, 0) + 1
        return counts
    
    def greedy_remove(S, mem, deadline, batch_size=3):
        """Remove elements to eliminate all bad triples."""
        bad = find_bad_triples(S, mem)
        while bad and time.time() < deadline:
            counts = count_involvement(bad, S)
            if not counts:
                break
            # Prefer removing sums over members, then by involvement count
            # Sums are less restrictive since they're only in the "sum" position
            def removal_priority(x):
                # Lower priority = removed first
                # Sums (x > n or x that's only a sum) get lower priority
                # Actually, let's use: prefer removing elements with lower involvement
                # But also prefer sums since they appear in fewer "member" roles
                return -counts[x]
            
            sorted_counts = sorted(counts.items(), key=lambda x: (x[0] > n, -x[1]))
            for x, _ in sorted_counts[:batch_size]:
                if x in S:
                    S.discard(x)
                    mem[x] = False
            bad = find_bad_triples(S, mem)
        return S, mem
    
    def try_add_back(S, mem, removed, deadline):
        """Try to re-add removed elements that no longer cause violations."""
        for x in removed:
            if x in S:
                continue
            if time.time() >= deadline:
                break
            # Tentatively add x
            S.add(x)
            mem[x] = True
            bad = find_bad_triples(S, mem)
            if not bad:
                # Keep it
                pass
            else:
                # Revert
                S.discard(x)
                mem[x] = False
        return S, mem
    
    best_size = 0
    best_set = set()
    
    # Try multiple strategies with random restarts
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() >= deadline:
            break
        
        if restart == 0:
            # First restart: full set, greedy removal
            S = set(range(1, n + 1))
            mem = [False] * (max_sum + 1)
            for x in S:
                mem[x] = True
            removed = []
        elif restart == 1:
            # Second: prefer keeping small numbers, remove large sums
            S = set(range(1, n + 1))
            mem = [False] * (max_sum + 1)
            for x in S:
                mem[x] = True
            removed = []
        elif restart == 2:
            # Third: start with only numbers <= 64, then try to add larger
            S = set(range(1, 65))
            mem = [False] * (max_sum + 1)
            for x in S:
                mem[x] = True
            removed = []
        elif restart == 3:
            # Fourth: random subset
            S = set(random.sample(range(1, n + 1), 90))
            mem = [False] * (max_sum + 1)
            for x in S:
                mem[x] = True
            removed = []
        else:
            # Fifth: keep small, add large selectively
            S = set(range(1, 50)) | set(range(100, 129))
            mem = [False] * (max_sum + 1)
            for x in S:
                mem[x] = True
            removed = []
        
        S, mem = greedy_remove(S, mem, deadline, batch_size=3)
        
        # Try to add back removed elements
        if removed:
            S, mem = try_add_back(S, mem, removed, deadline)
        
        # Final verification and cleanup
        changed = True
        while changed and time.time() < deadline:
            changed = False
            bad = find_bad_triples(S, mem)
            if not bad:
                break
            for (a, b, c) in bad:
                # Remove the element with lowest "cost" (prefer sums)
                candidates = [(ab, ac, bc, a, b, c)]
                ab, ac, bc = a + b, a + c, b + c
                candidates = [ab, ac, bc, a, b, c]
                for x in candidates:
                    if x in S:
                        S.discard(x)
                        mem[x] = False
                        changed = True
                        break
                if changed:
                    break
        
        # Update best
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # If best_set is empty, fall back to a simple valid set
    if not best_set:
        best_set = set(range(1, 65))
    
    return list(sorted(best_set))
