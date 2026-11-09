for x in range(1, 129):
    for color in range(128):
        valid = True
        # Check 4-APs where x is the last element
        for d in range(1, (x-1)//3 + 1):
            if x - 3*d < 1:
                continue
            c2 = colors[x-1-d]
            c3 = colors[x-1-2*d]
            c4 = colors[x-1-3*d]
            if len({color, c2, c3, c4}) < 3:
                valid = False
                break
        if not valid:
            continue
        # ...
        colors.append(color)
        break
