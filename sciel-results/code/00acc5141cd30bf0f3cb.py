            valid = True
            
            # Check 2*x
            if counts.get(2 * x, 0) >= 2:
                valid = False
            
            if valid:
                # Check x+y for all y in S
                for y in S:
                    s = x + y
                    if counts.get(s, 0) >= 2:
                        valid = False
                        break
            
            if valid:
                S.append(x)
                counts[2 * x] = counts.get(2 * x, 0) + 1
                for y in S[:-1]:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
