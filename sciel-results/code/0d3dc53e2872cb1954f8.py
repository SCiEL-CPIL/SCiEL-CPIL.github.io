import time

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    best_set = set()
    best_size = 0
    
    # L' = {1, ..., k}
    # Check if L' is 200-free.
    # For k <= 19, max sum is 190 < 200. So always 200-free.
    
    for k in range(0, 20):
        if time.time() > deadline:
            break
            
        L = set(range(1, k+1))
        
        # Determine forbidden ranges for H'
        # Pairs in H' summing to P
        # Triples in H' summing to T
        
        # P from 2H+1L: P = 200 - l, l in L
        # P from 2H+2L: P = 200 - (l1+l2), l1,l2 in L, l1!=l2
        # T from 3H+0L: T = 200
        # T from 3H+1L: T = 200 - l, l in L
        
        forbidden_pairs = set()
        forbidden_triples = set()
        
        # 2H+1L
        for l in L:
            forbidden_pairs.add(200 - l)
            
        # 2H+2L
        if len(L) >= 2:
            l_list = list(L)
            for i in range(len(l_list)):
                for j in range(i+1, len(l_list)):
                    forbidden_pairs.add(200 - (l_list[i] + l_list[j]))
                    
        # 3H+0L
        forbidden_triples.add(200)
        
        # 3H+1L
        for l in L:
            forbidden_triples.add(200 - l)
            
        # Filter forbidden values to relevant ranges for H'
        # H' elements are in [51, 100]
        # Min pair sum 51+52=103. Max 99+100=199.
        # Min triple sum 51+52+53=156. Max 98+99+100=297.
        
        # We only care about forbidden sums that are achievable by H'
        # But simpler to just check membership.
        
        H = set()
        # Iterate H candidates from 100 down to 51
        for h in range(100, 50, -1):
            if time.time() > deadline:
                break
                
            # Check pair constraints
            valid = True
            for h1 in H:
                s = h + h1
                if s in forbidden_pairs:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check triple constraints
            for h1 in H:
                for h2 in H:
                    if h1 >= h2:
                        continue
                    s = h + h1 + h2
                    if s in forbidden_triples:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                H.add(h)
                
        current_set = L.union(H)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    return list(best_set)
