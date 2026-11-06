def construct():
    n = 256
    S = []
    # Greedy from largest to smallest
    for a in range(n, 0, -1):
        # Check if a can be added to S
        # All b, c in S are > a
        # Condition: a does not divide 2b for any b in S
        # And a does not divide b+c for any distinct b, c in S
        
        valid = True
        
        # Check 2b
        for b in S:
            if (2 * b) % a == 0:
                valid = False
                break
        
        if valid:
            # Check b+c
            # S is small (max 256), so O(|S|^2) is fine
            len_S = len(S)
            for i in range(len_S):
                if not valid:
                    break
                for j in range(i + 1, len_S):
                    if (S[i] + S[j]) % a == 0:
                        valid = False
                        break
        
        if valid:
            S.append(a)
            
    # S contains the selected numbers
    # Return as sorted list
    return sorted(S)
