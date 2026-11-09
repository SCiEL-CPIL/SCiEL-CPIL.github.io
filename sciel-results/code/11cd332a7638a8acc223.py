def construct():
    answers = []
    for n in range(3, 257):
        found = None
        # x must be such that 1/x < 4/n, so x > n/4
        # and 3/x > 4/n (since 1/y+1/z < 2/x is not tight, but 1/y+1/z < 1/x+1/x=2/x? No, y,z > x so 1/y, 1/z < 1/x. Sum < 2/x.
        # 4/n < 1/x + 2/x = 3/x => x < 3n/4)
        start_x = n // 4 + 1
        end_x = (3 * n) // 4
        
        for x in range(start_x, end_x + 1):
            A = 4 * x - n
            B = n * x
            # y > x
            # z > y => By / (Ay - B) > y => B > Ay - B => 2B > Ay => y < 2B/A
            # So y is in [x+1, floor(2B/A)]
            max_y = (2 * B) // A
            if max_y <= x:
                continue
                
            for y in range(x + 1, max_y + 1):
                D = A * y - B
                if D <= 0:
                    continue
                if (B * y) % D == 0:
                    z = (B * y) // D
                    if z > y:
                        found = [x, y, z]
                        break
            if found:
                break
        answers.append(found)
    return answers
