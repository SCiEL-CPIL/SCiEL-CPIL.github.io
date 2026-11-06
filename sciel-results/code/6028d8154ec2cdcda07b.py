    def is_bad(ca, cb, cc, cd):
        # Returns True if distinct count < 3
        # Distinct < 3 means 1 or 2 distinct colors.
        if ca == cb == cc == cd: return True
        if ca == cb == cc: return True
        if ca == cb == cd: return True
        if ca == cc == cd: return True
        if cb == cc == cd: return True
        if ca == cb and cc == cd: return True
        if ca == cc and cb == cd: return True
        if ca == cd and cb == cc: return True
        return False
    