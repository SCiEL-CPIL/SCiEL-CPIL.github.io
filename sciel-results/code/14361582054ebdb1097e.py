        diffs_A = set()
        A_list = list(A_set)
        for i in range(len(A_list)):
            for j in range(len(A_list)):
                if i != j:
                    diffs_A.add(abs(A_list[i] - A_list[j]))
