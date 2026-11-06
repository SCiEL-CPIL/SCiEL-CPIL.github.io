def priority(vector, n):
        count_2 = 0
        for x in vector:
            if x == 2:
                count_2 += 1
        if count_2 == 1:
            return 1.0
        return 0.0
