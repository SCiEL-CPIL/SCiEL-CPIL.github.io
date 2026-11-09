    def creates_q3(u, v):
        remaining = [x for x in all_vertices if x != u and x != v]
        
        for s6 in itertools.combinations(remaining, 6):
            if time.time() > deadline:
                return True
            
            # s6 is a tuple of 6
            # We need to split s6 into two sets of 3.
            # One set joins u in A, the other joins v in B.
            # C(6,3) = 20
            for a_others in itertools.combinations(s6, 3):
                # a_others is tuple of 3
                # A = {u} U a_others
                # B = {v} U (s6 \ a_others)
                
                # Construct B set
                # Since s6 is small, we can just iterate
                b_others = tuple(x for x in s6 if x not in a_others)
                
                # Check missing edges
                # A vertices: u, a1, a2, a3
                # B vertices: v, b1, b2, b3
                
                # Mask for B
                mask_B = (1 << v)
                for b in b_others:
                    mask_B |= (1 << b)
                
                valid = True
                missing_bs = []
                
                # Check u
                # u is in A
                neigh_u_in_B = adj[u] & mask_B
                missing_u_in_B = mask_B ^ neigh_u_in_B
                if missing_u_in_B.bit_count() != 1:
                    valid = False
                else:
                    b = missing_u_in_B.bit_length() - 1
                    missing_bs.append(b)
                
                if valid:
                    # Check a1, a2, a3
                    for a in a_others:
                        neigh_a_in_B = adj[a] & mask_B
                        missing_a_in_B = mask_B ^ neigh_a_in_B
                        if missing_a_in_B.bit_count() != 1:
                            valid = False
                            break
                        b = missing_a_in_B.bit_length() - 1
                        if b in missing_bs:
                            valid = False
                            break
                        missing_bs.append(b)
                        
                if valid:
                    return True
                    
        return False
