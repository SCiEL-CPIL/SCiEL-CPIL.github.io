def construct():
    answers = []
    bound = 10**12
    for n in range(3, 257):
        found = None
        # x must satisfy 1/x < 4/n < 3/x  =>  n/4 < x < 3n/4
        # So x ranges from floor(n/4) + 1 to floor((3n-1)/4)
        x_min = n // 4 + 1
        x_max = (3 * n - 1) // 4
        
        # Iterate x from large to small to minimize the search range for y
        for x in range(x_max, x_min - 1, -1):
            A = 4 * x - n
            B = n * x
            if A <= 0:
                continue
            
            # y must satisfy y > B/A and y < 2B/A
            # Also y > x for distinctness (assuming x < y < z)
            y_min = max(x + 1, B // A + 1)
            y_max = (2 * B) // A
            
            if y_min > y_max:
                continue
                
            # Iterate y from large to small to minimize z
            # z = B*y / (A*y - B)
            # As y increases, z decreases.
            # So start from y_max to get smallest z.
            for y in range(y_max, y_min - 1, -1):
                denom = A * y - B
                if denom <= 0:
                    continue
                # Check if z is integer
                if (B * y) % denom == 0:
                    z = (B * y) // denom
                    if z > 10**12:
                        # Since z decreases as y increases, and we are going down y,
                        # z will increase. So if current z > bound, all smaller y will have larger z.
                        # Thus, no solution for this x.
                        break
                    if z > y:
                        found = [x, y, z]
                        break
            if found:
                break
        
        answers.append(found)
    return answers
