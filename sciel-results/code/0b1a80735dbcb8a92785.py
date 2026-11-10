    def is_valid_solution(rows):
        global_mask = 0
        for r in rows:
            m = row_to_mask(r)
            if bin(m).count('1') != len(r)*(len(r)-1)//2:
                return False
            if global_mask & m:
                return False
            global_mask |= m
        return True
