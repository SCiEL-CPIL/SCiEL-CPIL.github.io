        for a in smooth:
            if a >= n:
                break
            b = n - a
            if b <= a:
                continue
            if b in smooth_set:
                # Check antichain: a does not divide b
                if b % a != 0:
                    answers.append([a, b])
                    found = True
                    break
