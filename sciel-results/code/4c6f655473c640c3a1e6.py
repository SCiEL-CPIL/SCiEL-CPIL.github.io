import time
import random

def construct():
    n = 128
    # A is a set of integers
    A = set(range(1, n + 1))
    
    def get_bad_triples(A):
        bads = []
        # Optimization: only check if a+b <= n, etc.
        # But sums can be up to 2n. If sum > n, it's not in A (since A subset 1..n).
        # So we only care if sums are in 1..n.
        # a,b,c distinct.
        # a+b, a+c, b+c must be in A.
        # Since a,b,c >= 1, sums >= 2.
        # If a+b > n, it's not in A.
        # So we only need to check triples where a+b <= n, a+c <= n, b+c <= n.
        # This implies a,b,c are relatively small.
        # Specifically, if we sort a<b<c, then b+c <= n is the strongest condition.
        # So c <= n - b.
        
        # To speed up, iterate c from 1 to n//2?
        # If b >= n/2, b+c > n for any c>b.
        # So b < n/2.
        # a < b < n/2.
        
        elems = sorted(A)
        # Map value to index? Or just use set membership.
        # Set membership is O(1).
        
        # We can just iterate all triples? 128 choose 3 is ~340k.
        # 340k * 6 checks = 2M operations. Very fast.
        
        for i in range(len(elems)):
            a = elems[i]
            for j in range(i+1, len(elems)):
                b = elems[j]
                if a + b > n: break # since c>b, a+c > a+b > n
                for k in range(j+1, len(elems)):
                    c = elems[k]
                    if b + c > n: break # a+c < b+c, so if b+c>n, no need to check further c? 
                    # Wait, a+c < b+c. If b+c > n, then a+c might be <= n.
                    # But we need ALL sums in A.
                    # If b+c > n, then b+c not in A. So triple is not bad.
                    # So we can break if b+c > n?
                    # Yes, because c is increasing.
                    
                    if a + c in A and b + c in A:
                        # a+b is already known to be in A? No, we didn't check a+b in A.
                        # We need a+b in A too.
                        if a + b in A:
                            bads.append((a, b, c))
        return bads

    # Heuristic: Remove element in most bad triples
    # To avoid O(N^3) repeatedly, we can just do it a few times.
    # Or use a smarter update.
    
    # Let's just do a simple local search.
    # Start with full set.
    # While bad triples exist:
    #   Pick a random bad triple.
    #   Remove one of its elements.
    
    # This might be slow if we do it one by one.
    # Better: Find all bad triples, count involvement, remove max.
    
    start_time = time.time()
    while time.time() - start_time < 80:
        bads = get_bad_triples(A)
        if not bads:
            break
        
        # Count involvement
        counts = {}
        for a, b, c in bads:
            for x in [a, b, c, a+b, a+c, b+c]:
                if x in A:
                    counts[x] = counts.get(x, 0) + 1
        
        # Remove element with max count
        if counts:
            max_elem = max(counts, key=counts.get)
            A.remove(max_elem)
        else:
            break
            
    # Now try to add elements back
    # Sort candidates by potential
    candidates = [x for x in range(1, n + 1) if x not in A]
    random.shuffle(candidates)
    
    for x in candidates:
        if time.time() - start_time > 85:
            break
        # Try adding x
        A.add(x)
        bads = get_bad_triples(A)
        if not bads:
            continue # Keep x
        else:
            # Check if we can remove one element to fix
            # Find a bad triple involving x
            # If we can remove y such that all bads are gone, do it?
            # This is complex. Just revert for now.
            A.remove(x)
            
    return list(A)
