    def reduce_solution(codewords):
        c_list = codewords[:]
        random.shuffle(c_list)
        new_list = []
        for c in c_list:
            covered_by_others = 0
            for o in new_list:
                covered_by_others |= sphere_masks[o]
            unique_to_c = sphere_masks[c] & (~covered_by_others)
            if unique_to_c == 0:
                continue
            else:
                new_list.append(c)
        
        # Loop until no change
        while True:
            old_size = len(new_list)
            c_list = new_list[:]
            random.shuffle(c_list)
            final_list = []
            for c in c_list:
                covered_by_others = 0
                for o in final_list:
                    covered_by_others |= sphere_masks[o]
                unique_to_c = sphere_masks[c] & (~covered_by_others)
                if unique_to_c == 0:
                    continue
                else:
                    final_list.append(c)
            if len(final_list) == old_size:
                return final_list
            new_list = final_list
