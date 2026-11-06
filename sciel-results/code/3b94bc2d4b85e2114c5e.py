    def can_add(S_set, x):
        """Check if x can be added to set S (represented as a set of values)."""
        # Check condition 1: for each a in S, no pair (b,c) in S\{a} with (b+c+x) % a == 0
        for a in S_set:
            if a == x:
                continue
            target = (-x) % a
            # Count how many pairs (b,c) in S\{a} have (b+c) % a == target
            # We need at least one such pair where b != c and b,c != a
            # Use residue counting
            residue_count = {}
            for v in S_set:
                if v == a:
                    continue
                r = v % a
                residue_count[r] = residue_count.get(r, 0) + 1
            
            # Count pairs (b,c) with b != c, b,c != a, (b+c) % a == target
            # Iterate over residues
            found = False
            for r1 in residue_count:
                r2 = (target - r1) % a
                if r2 not in residue_count:
                    continue
                cnt1 = residue_count[r1]
                cnt2 = residue_count[r2]
                if r1 == r2:
                    # Need at least 2 distinct elements with this residue
                    if cnt1 >= 2:
                        found = True
                        break
                else:
                    # Need at least 1 from each residue
                    if cnt1 >= 1 and cnt2 >= 1:
                        found = True
                        break
            if found:
                return False
        
        # Check condition 2: no triple (b,c,d) in S with (b+c+d) % x == 0
        if len(S_set) >= 3:
            # Precompute residue counts for S mod x
            residue_count_x = {}
            S_list = list(S_set)
            for v in S_list:
                r = v % x
                residue_count_x[r] = residue_count_x.get(r, 0) + 1
            
            for i in range(len(S_list)):
                b = S_list[i]
                for j in range(i + 1, len(S_list)):
                    c = S_list[j]
                    target = (-b - c) % x
                    # Check if any d in S\{b,c} has d % x == target
                    # Count of d with residue target
                    cnt_target = residue_count_x.get(target, 0)
                    # Subtract b and c if they have this residue
                    if b % x == target:
                        cnt_target -= 1
                    if c % x == target:
                        cnt_target -= 1
                    if cnt_target > 0:
                        return False
        return True
